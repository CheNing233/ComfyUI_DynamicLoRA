"""Dynamic per-denoise-step rank and strength control for native ComfyUI LoRAs."""

from __future__ import annotations

import logging
import threading
import uuid
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
    from .schedule import parse_ratio_schedule, ratio_for_step, step_index_for_sigma, active_rank
except ImportError:
    from schedule import parse_ratio_schedule, ratio_for_step, step_index_for_sigma, active_rank


LOGGER = logging.getLogger("ComfyUI-DynamicLoraRank")
_INJECTION_KEY_PREFIX = "dynamic_lora_rank_bypass"
_WRAPPER_KEY_PREFIX = "dynamic_lora_rank_step_wrapper"
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

    def __init__(self, source: LoRAAdapter, rank_schedule, strength_schedule):
        self.loaded_keys = source.loaded_keys
        self.weights = source.weights
        self.rank_schedule = tuple(rank_schedule)
        self.strength_schedule = tuple(strength_schedule)

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
        scale *= strength_ratio

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


def _new_scheduled_adapter(source: LoRAAdapter, rank_schedule, strength_schedule):
    return ScheduledLoRAAdapter(source, rank_schedule, strength_schedule)


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
        scheduled_patches[key] = _new_scheduled_adapter(
            patch_data,
            rank_schedule,
            strength_schedule,
        )
    return scheduled_patches, regular_patches


def _add_model_step_wrapper(model_patcher) -> None:
    key = f"{_WRAPPER_KEY_PREFIX}_{uuid.uuid4().hex}"
    model_patcher.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.PREDICT_NOISE,
        key,
        _predict_noise_step_wrapper,
    )


def _load_bypass_with_schedule(
    model,
    clip,
    lora,
    strength_model,
    strength_clip,
    rank_schedule,
    strength_schedule,
):
    key_map = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

    loaded = comfy.lora.load_lora(comfy.lora_convert.convert_lora(lora), key_map)
    scheduled_patches, regular_patches = _classify_loaded_patches(
        loaded,
        rank_schedule,
        strength_schedule,
    )

    model_out = None
    clip_out = None

    if model is not None:
        model_out = model.clone()
        if regular_patches:
            model_out.add_patches(regular_patches, strength_model)

        manager = comfy.weight_adapter.BypassInjectionManager()
        model_keys = set(model_out.model.state_dict())
        for key, adapter in scheduled_patches.items():
            if key in model_keys:
                manager.add_adapter(
                    key,
                    _new_scheduled_adapter(
                        adapter,
                        adapter.rank_schedule,
                        adapter.strength_schedule,
                    ),
                    strength=strength_model,
                )
            else:
                LOGGER.warning("Dynamic LoRA key %r was not found in the model; skipping model bypass injection.", key)

        injections = manager.create_injections(model_out.model)
        if manager.get_hook_count() > 0:
            model_out.set_injections(f"{_INJECTION_KEY_PREFIX}_{uuid.uuid4().hex}", injections)
            _add_model_step_wrapper(model_out)

    if clip is not None:
        clip_out = clip.clone()
        if regular_patches:
            clip_out.add_patches(regular_patches, strength_clip)

        manager = comfy.weight_adapter.BypassInjectionManager()
        clip_keys = set(clip_out.cond_stage_model.state_dict())
        for key, adapter in scheduled_patches.items():
            if key in clip_keys:
                # No sampler step exists during CLIP encoding, so the adapter
                # naturally uses full rank and dynamic strength 1.0.
                manager.add_adapter(
                    key,
                    _new_scheduled_adapter(
                        adapter,
                        adapter.rank_schedule,
                        adapter.strength_schedule,
                    ),
                    strength=strength_clip,
                )
            else:
                LOGGER.debug("Dynamic LoRA key %r is not a CLIP key; skipping CLIP bypass injection.", key)

        injections = manager.create_injections(clip_out.cond_stage_model)
        if manager.get_hook_count() > 0:
            clip_out.patcher.set_injections(f"{_INJECTION_KEY_PREFIX}_{uuid.uuid4().hex}", injections)

    for key in loaded:
        if key not in scheduled_patches and key not in regular_patches:
            LOGGER.warning("LoRA key %r was not loaded by the dynamic loader.", key)

    return model_out, clip_out


class DynamicLoraRankLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model to modify."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model to modify."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The native ComfyUI LoRA file."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "rank_schedule": ("STRING", {"default": "1.0", "multiline": False, "tooltip": "Per-step rank ratios, e.g. 1,0.5,1,0. Missing steps repeat the last value."}),
                "strength_schedule": ("STRING", {"default": "1.0", "multiline": False, "tooltip": "Per-step strength multipliers relative to strength_model, e.g. 1,0.5,1,0."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    CATEGORY = "model/loaders"
    DESCRIPTION = "Native LoRA loader with per-denoise-step rank and strength schedules."
    SEARCH_ALIASES = ["dynamic lora rank", "scheduled lora", "t lora", "rank lora"]

    def __init__(self):
        self.loaded_lora = None

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip, rank_schedule, strength_schedule="1.0"):
        if strength_model == 0 and strength_clip == 0:
            return model, clip
        rank_values = parse_ratio_schedule(rank_schedule, "rank_schedule")
        strength_values = parse_ratio_schedule(strength_schedule, "strength_schedule")
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        if self.loaded_lora is None or self.loaded_lora[0] != lora_path:
            lora, metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
            self.loaded_lora = (lora_path, lora, metadata)
        else:
            lora, metadata = self.loaded_lora[1], self.loaded_lora[2]

        model_out, clip_out = _load_bypass_with_schedule(
            model,
            clip,
            lora,
            strength_model,
            strength_clip,
            rank_values,
            strength_values,
        )
        if metadata:
            if model_out is not None:
                model_out.set_attachments("lora_metadata", metadata)
            if clip_out is not None:
                clip_out.patcher.set_attachments("lora_metadata", metadata)
        return model_out, clip_out


class DynamicLoraRankLoaderModelOnly(DynamicLoraRankLoader):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "rank_schedule": ("STRING", {"default": "1.0", "multiline": False}),
                "strength_schedule": ("STRING", {"default": "1.0", "multiline": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model_only"

    def load_model_only(self, model, lora_name, strength_model, rank_schedule, strength_schedule="1.0"):
        model_out, _ = self.load_lora(
            model,
            None,
            lora_name,
            strength_model,
            0.0,
            rank_schedule,
            strength_schedule,
        )
        return (model_out,)


NODE_CLASS_MAPPINGS = {
    "DynamicLoraRankLoader": DynamicLoraRankLoader,
    "DynamicLoraRankLoaderModelOnly": DynamicLoraRankLoaderModelOnly,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DynamicLoraRankLoader": "Load LoRA (Dynamic Rank + Strength)",
    "DynamicLoraRankLoaderModelOnly": "Load LoRA (Dynamic Rank + Strength, Model Only)",
}
