"""Dynamic per-denoise-step rank and strength control for native ComfyUI LoRAs."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import folder_paths
import torch
import torch.nn.functional as F

import comfy.lora
import comfy.lora_convert
import comfy.patcher_extension
import comfy.utils
import comfy.weight_adapter
from comfy.weight_adapter.lora import LoRAAdapter

try:
    from .schedule import parse_ratio_schedule, parse_strength_schedule, ratio_for_step, step_index_for_sigma, active_rank
except ImportError:
    from schedule import parse_ratio_schedule, parse_strength_schedule, ratio_for_step, step_index_for_sigma, active_rank
try:
    from .waveform import (
        MONOTONIC_CURVES,
        OSCILLATION_WAVEFORMS,
        format_schedule,
        generate_monotonic_schedule,
        generate_oscillation_schedule,
        flow_shift_schedule,
        parse_numeric_schedule,
        render_schedule_preview,
    )
except ImportError:
    from waveform import (
        MONOTONIC_CURVES,
        OSCILLATION_WAVEFORMS,
        format_schedule,
        generate_monotonic_schedule,
        generate_oscillation_schedule,
        flow_shift_schedule,
        parse_numeric_schedule,
        render_schedule_preview,
    )


LOGGER = logging.getLogger("ComfyUI-DynamicLoraRank")
_INJECTION_KEY = "dynamic_lora_rank_bypass"
_WRAPPER_KEY = "dynamic_lora_rank_step_wrapper"
_REGISTRY_ATTACHMENT = "dynamic_lora_rank_registry"
_STEP_STATE = threading.local()
_MISSING = object()


def _set_current_step(step_index: int):
    previous = getattr(_STEP_STATE, "step_index", _MISSING)
    _STEP_STATE.step_index = int(step_index)
    return previous


def _restore_current_step(previous) -> None:
    if previous is _MISSING:
        if hasattr(_STEP_STATE, "step_index"):
            delattr(_STEP_STATE, "step_index")
    else:
        _STEP_STATE.step_index = previous


def _get_current_step() -> int | None:
    value = getattr(_STEP_STATE, "step_index", None)
    return None if value is None else int(value)


def _rank_mask(
    hidden: torch.Tensor,
    rank: int,
    ratio: float | None,
    is_conv: bool,
) -> torch.Tensor | None:
    """Create a rank mask on the correct axis for Linear or Conv outputs."""
    if ratio is None:
        return None

    count = active_rank(rank, ratio)
    if is_conv:
        # Conv output layout: [batch, rank, spatial...].
        shape = [1, rank] + [1] * max(0, hidden.ndim - 2)
        rank_axis = 1
    else:
        # Linear output layout: [..., rank], including Anima [B, S, rank].
        shape = [1] * hidden.ndim
        shape[-1] = rank
        rank_axis = hidden.ndim - 1

    mask = torch.zeros(shape, device=hidden.device, dtype=hidden.dtype)
    if count:
        index = [slice(None)] * hidden.ndim
        index[rank_axis] = slice(0, count)
        mask[tuple(index)] = 1.0
    return mask


class ScheduledLoRAAdapter(LoRAAdapter):
    """Standard native LoRA adapter with per-step rank and strength controls."""

    def __init__(self, source: LoRAAdapter, rank_schedule, strength_schedule, base_strength: float = 1.0):
        self.loaded_keys = source.loaded_keys
        self.weights = source.weights
        self.rank_schedule = tuple(rank_schedule)
        self.strength_schedule = tuple(strength_schedule)
        self.base_strength = float(base_strength)

    def _schedule_values(self) -> tuple[float | None, float]:
        step_index = _get_current_step()
        if step_index is None:
            # CLIP encoding and non-sampling use preserve native static strength
            # and full rank behavior.
            return None, 1.0
        return (
            ratio_for_step(self.rank_schedule, step_index),
            ratio_for_step(self.strength_schedule, step_index),
        )

    def h(self, x: torch.Tensor, _base_out: torch.Tensor) -> torch.Tensor:
        # This mirrors ComfyUI's native LoRAAdapter.h, with a rank mask after
        # the down projection and a scalar multiplier on the complete output.
        func_list = [None, None, F.linear, F.conv1d, F.conv2d, F.conv3d]
        up, down, alpha, mid = self.weights[0], self.weights[1], self.weights[2], self.weights[3]
        rank = int(down.shape[0])
        scale = (alpha / rank) if alpha is not None else 1.0
        scale *= getattr(self, "multiplier", 1.0)
        rank_ratio, strength_ratio = self._schedule_values()
        scale *= self.base_strength * strength_ratio

        orig_dtype = x.dtype
        up = up.to(dtype=x.dtype, device=x.device)
        down = down.to(dtype=x.dtype, device=x.device)
        mid = mid.to(dtype=x.dtype, device=x.device) if mid is not None else None

        is_conv = bool(getattr(self, "is_conv", False))
        conv_dim = int(getattr(self, "conv_dim", 0))
        kw_dict = getattr(self, "kw_dict", {})
        if is_conv:
            op = func_list[conv_dim + 2]
            if down.dim() == 2:
                down = down.view(
                    down.shape[0],
                    getattr(self, "in_channels", down.shape[1]),
                    *getattr(self, "kernel_size", (1,) * conv_dim),
                )
            if up.dim() == 2:
                up = up.view(*up.shape, *([1] * conv_dim))
            hidden = op(x, down, **kw_dict)
        else:
            hidden = F.linear(x, down)

        mask = _rank_mask(hidden, rank, rank_ratio, is_conv)
        if mask is not None:
            hidden = hidden * mask

        if mid is not None:
            if is_conv:
                if mid.dim() == 2:
                    mid = mid.view(*mid.shape, *([1] * conv_dim))
                hidden = op(hidden, mid, **kw_dict)
            else:
                hidden = F.linear(hidden, mid)

        if is_conv:
            out = op(hidden, up)
        else:
            out = F.linear(hidden, up)
        return out.to(orig_dtype) * scale


def _step_index_for_wrapper(timestep: torch.Tensor, model_options: dict[str, Any]) -> int:
    transformer_options = model_options.get("transformer_options", {}) if model_options else {}
    sample_sigmas = transformer_options.get("sample_sigmas")
    if isinstance(timestep, torch.Tensor) and timestep.numel():
        sigma = float(timestep.reshape(-1)[0].detach().float().cpu())
    else:
        sigma = 0.0
    return step_index_for_sigma(sigma, sample_sigmas)


def _predict_noise_step_wrapper(executor, x, timestep, model_options=None, seed=None):
    """Expose only the current step index; each adapter owns its own schedule."""
    model_options = model_options or {}
    previous = _set_current_step(_step_index_for_wrapper(timestep, model_options))
    try:
        return executor(x, timestep, model_options=model_options, seed=seed)
    finally:
        _restore_current_step(previous)


class CompositeScheduledAdapter(comfy.weight_adapter.WeightAdapterBase):
    """Combines multiple standard LoRAs into one hook for a target module."""

    def __init__(self, adapters):
        self.adapters = list(adapters)
        self.loaded_keys = set()
        for adapter in self.adapters:
            self.loaded_keys.update(adapter.loaded_keys)
        self.weights = ()

    def _sync_runtime_attributes(self):
        for adapter in self.adapters:
            adapter.multiplier = 1.0
            adapter.is_conv = self.is_conv
            adapter.conv_dim = self.conv_dim
            adapter.kernel_size = self.kernel_size
            adapter.in_channels = self.in_channels
            adapter.out_channels = self.out_channels
            adapter.kw_dict = self.kw_dict

    def h(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        self._sync_runtime_attributes()
        output = torch.zeros_like(base_out)
        for adapter in self.adapters:
            output = output + adapter.h(x, base_out)
        return output

    def g(self, y: torch.Tensor) -> torch.Tensor:
        return y


def _registry_adapter(source: LoRAAdapter, rank_schedule, strength_schedule, base_strength: float = 1.0):
    if isinstance(source, ScheduledLoRAAdapter):
        return source
    return ScheduledLoRAAdapter(source, rank_schedule, strength_schedule, base_strength)


def _copy_registry(registry):
    return {key: list(adapters) for key, adapters in (registry or {}).items()}


def _patch_key_exists(keys, key) -> bool:
    return key in keys or (isinstance(key, str) and f"{key}.weight" in keys)


def _classify_loaded_patches(loaded: dict, rank_schedule, strength_schedule):
    scheduled_patches = {}
    regular_patches = {}
    for key, patch_data in loaded.items():
        if not isinstance(patch_data, comfy.weight_adapter.WeightAdapterBase):
            regular_patches[key] = patch_data
            continue

        # The bypass implementation below is exact for standard LoRAAdapter
        # weights. DoRA/reshape and other adapter families use native static
        # patching until their forward math is implemented explicitly.
        if not isinstance(patch_data, LoRAAdapter):
            LOGGER.warning(
                "Dynamic schedules are not supported for adapter %s at key %r; using native static patching.",
                type(patch_data).__name__,
                key,
            )
            regular_patches[key] = patch_data
            continue
        if patch_data.weights[4] is not None or patch_data.weights[5] is not None:
            LOGGER.warning(
                "Dynamic schedules are not supported for LoRA metadata (DoRA/reshape) at key %r; using native static patching.",
                key,
            )
            regular_patches[key] = patch_data
            continue
        if not isinstance(key, str):
            LOGGER.warning(
                "Sliced/tuple LoRA mapping %r cannot use dynamic bypass; using native static patching.",
                key,
            )
            regular_patches[key] = patch_data
            continue
        scheduled_patches[key] = _registry_adapter(
            patch_data,
            rank_schedule,
            strength_schedule,
        )
    return scheduled_patches, regular_patches


def _add_model_step_wrapper(model_patcher) -> None:
    wrappers = model_patcher.get_wrappers(
        comfy.patcher_extension.WrappersMP.PREDICT_NOISE,
        _WRAPPER_KEY,
    )
    if not wrappers:
        model_patcher.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.PREDICT_NOISE,
            _WRAPPER_KEY,
            _predict_noise_step_wrapper,
        )


def _rebuild_dynamic_injections(patcher, registry, is_clip=False):
    """Replace the plugin's single injection group without stacking hooks."""
    if patcher.is_injected and _INJECTION_KEY in patcher.injections:
        patcher.eject_model()
    patcher.remove_injections(_INJECTION_KEY)
    patcher.set_attachments(_REGISTRY_ATTACHMENT, registry)

    if not registry:
        return

    manager = comfy.weight_adapter.BypassInjectionManager()
    model_keys = set(patcher.model.state_dict())
    for key, adapters in registry.items():
        key_exists = _patch_key_exists(model_keys, key)
        if not key_exists:
            LOGGER.warning("Dynamic LoRA key %r was not found in the %s model; skipping injection.", key, "CLIP" if is_clip else "diffusion")
            continue
        manager.add_adapter(key, CompositeScheduledAdapter(adapters), strength=1.0)

    target = patcher.model
    injections = manager.create_injections(target)
    if manager.get_hook_count() > 0:
        patcher.set_injections(_INJECTION_KEY, injections)
        if not is_clip:
            _add_model_step_wrapper(patcher)


def _load_bypass_with_schedule(model, lora, rank_schedule, strength_schedule):
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    loaded = comfy.lora.load_lora(comfy.lora_convert.convert_lora(lora), key_map)
    static_strength = float(strength_schedule[0]) if strength_schedule else 1.0
    scheduled_patches, regular_patches = _classify_loaded_patches(
        loaded,
        rank_schedule,
        strength_schedule,
    )

    model_out = model.clone()
    if regular_patches:
        # Unsupported adapter families retain native math with the first
        # configured strength as their static fallback.
        model_out.add_patches(regular_patches, static_strength)

    registry = _copy_registry(model.get_attachment(_REGISTRY_ATTACHMENT))
    model_keys = set(model_out.model.state_dict())
    for key, adapter in scheduled_patches.items():
        if _patch_key_exists(model_keys, key):
            registry.setdefault(key, []).append(
                _registry_adapter(
                    adapter,
                    adapter.rank_schedule,
                    adapter.strength_schedule,
                    1.0,
                )
            )
    _rebuild_dynamic_injections(model_out, registry, is_clip=False)

    for key in loaded:
        if key not in scheduled_patches and key not in regular_patches:
            LOGGER.warning("LoRA key %r was not loaded by the dynamic loader.", key)

    return model_out


class XCN_DynamicLoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model to modify."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The native ComfyUI LoRA file."}),
                "rank_schedule": ("STRING", {"default": "1.0", "multiline": False, "tooltip": "Per-step rank ratios, e.g. 1,0.5,1,0."}),
                "strength_schedule": ("STRING", {"default": "1.0", "multiline": False, "tooltip": "Absolute per-step LoRA strengths, e.g. 1,0.5,1,0."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model_only"
    CATEGORY = "model/loaders"
    DESCRIPTION = "Native model-only LoRA loader with per-denoise-step rank and absolute strength schedules."
    SEARCH_ALIASES = ["dynamic lora", "dynamic lora rank", "scheduled lora", "t lora"]

    def __init__(self):
        self.loaded_lora = None

    def load_model_only(self, model, lora_name, rank_schedule, strength_schedule="1.0"):
        rank_values = parse_ratio_schedule(rank_schedule, "rank_schedule")
        strength_values = parse_strength_schedule(strength_schedule)
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        if self.loaded_lora is None or self.loaded_lora[0] != lora_path:
            lora, metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
            self.loaded_lora = (lora_path, lora, metadata)
        else:
            lora, metadata = self.loaded_lora[1], self.loaded_lora[2]

        model_out = _load_bypass_with_schedule(
            model,
            lora,
            rank_values,
            strength_values,
        )
        if metadata:
            model_out.set_attachments("lora_metadata", metadata)
        return (model_out,)


class XCN_OscillationSchedule:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 1, "max": 8192, "step": 1}),
                "waveform": (list(OSCILLATION_WAVEFORMS), {"default": "cosine"}),
                "x_step_offset": ("FLOAT", {"default": 0.0, "min": -8192.0, "max": 8192.0, "step": 0.1}),
                "y_offset": ("FLOAT", {"default": 0.5, "min": -100.0, "max": 100.0, "step": 0.01}),
                "amplitude": ("FLOAT", {"default": 0.5, "min": -100.0, "max": 100.0, "step": 0.01}),
                "cycles": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 256.0, "step": 0.1}),
                "min_value": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "max_value": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "start_step": ("INT", {"default": 1, "min": 1, "max": 8192, "step": 1}),
                "end_step": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1, "tooltip": "0 means the last step."}),
                "decimals": ("INT", {"default": 6, "min": 0, "max": 12, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("schedule",)
    FUNCTION = "generate"
    CATEGORY = "XCN/Schedule"
    DESCRIPTION = "Generate a square or cosine oscillation schedule using x cycles, y offset, amplitude, clipping, and an active step window."

    def generate(self, steps, waveform, x_step_offset, y_offset, amplitude, cycles, min_value, max_value, start_step, end_step, decimals):
        values = generate_oscillation_schedule(
            steps, waveform, x_step_offset, y_offset, amplitude, cycles, min_value, max_value,
            start_step, end_step, decimals,
        )
        return (format_schedule(values, decimals),)


class XCN_MonotonicSchedule:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 1, "max": 8192, "step": 1}),
                "curve": (list(MONOTONIC_CURVES), {"default": "cosine"}),
                "left_value": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "right_value": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "start_step": ("INT", {"default": 1, "min": 1, "max": 8192, "step": 1}),
                "end_step": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1, "tooltip": "0 means the last step."}),
                "decimals": ("INT", {"default": 6, "min": 0, "max": 12, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("schedule",)
    FUNCTION = "generate"
    CATEGORY = "XCN/Schedule"
    DESCRIPTION = "Generate a linear, cosine, exponential, or logarithmic monotonic schedule with left/right limits and an active step window."

    def generate(self, steps, curve, left_value, right_value, start_step, end_step, decimals):
        values = generate_monotonic_schedule(
            steps, curve, left_value, right_value, start_step, end_step, decimals,
        )
        return (format_schedule(values, decimals),)


class XCN_FlowShiftSchedule:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule": ("STRING", {"default": "1,0.5,1,0", "multiline": False}),
                "flow_shift": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01, "tooltip": "Anima uses shift=3.0 by default."}),
                "invert": ("BOOLEAN", {"default": True, "tooltip": "Pull values toward low-noise steps. Recommended for compensating Anima's native shift."}),
                "decimals": ("INT", {"default": 6, "min": 0, "max": 12, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("schedule",)
    FUNCTION = "remap"
    CATEGORY = "XCN/Schedule"
    DESCRIPTION = "Remap a discrete schedule with Anima flow shift, pulling values toward high-noise steps."

    def remap(self, schedule, flow_shift, invert, decimals):
        values = parse_numeric_schedule(schedule)
        return (format_schedule(flow_shift_schedule(values, flow_shift, invert, decimals), decimals),)


class XCN_SchedulePreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule": ("STRING", {"default": "1,0.5,1,0", "multiline": False}),
                "decimals": ("INT", {"default": 6, "min": 0, "max": 12, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("chart",)
    FUNCTION = "preview"
    CATEGORY = "XCN/Schedule"
    OUTPUT_NODE = True
    DESCRIPTION = "Render one chart with the smooth curve and every discrete step overlaid on the same axes."

    def preview(self, schedule, decimals):
        try:
            values = parse_numeric_schedule(schedule)
            image = render_schedule_preview(values, decimals)
            import numpy as np
            import torch
            array = np.asarray(image).astype(np.float32) / 255.0
            tensor = torch.from_numpy(array)[None, ...]
            temp_dir = folder_paths.get_temp_directory()
            full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                "XCN_SchedulePreview", temp_dir, image.width, image.height
            )
            file = f"{filename}_{counter:05}_.png"
            image.save(os.path.join(full_output_folder, file))
            ui_image = {"filename": file, "subfolder": subfolder, "type": "temp"}
            return {"ui": {"images": [ui_image]}, "result": (tensor,)}
        except ValueError as exc:
            import torch
            blank = torch.ones((1, 96, 640, 3), dtype=torch.float32)
            return {"ui": {"text": [f"解析失败：{exc}"]}, "result": (blank,)}


NODE_CLASS_MAPPINGS = {
    "XCN_DynamicLoraLoader": XCN_DynamicLoraLoader,
    "XCN_OscillationSchedule": XCN_OscillationSchedule,
    "XCN_MonotonicSchedule": XCN_MonotonicSchedule,
    "XCN_FlowShiftSchedule": XCN_FlowShiftSchedule,
    "XCN_SchedulePreview": XCN_SchedulePreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XCN_DynamicLoraLoader": "XCN Dynamic LoRA",
    "XCN_OscillationSchedule": "XCN Oscillation Schedule",
    "XCN_MonotonicSchedule": "XCN Monotonic Schedule",
    "XCN_FlowShiftSchedule": "XCN Flow Shift Schedule",
    "XCN_SchedulePreview": "XCN Schedule Preview",
}
