# Design Notes

## Runtime model

The loader keeps ComfyUI's native key mapping and file conversion, but sends only standard `LoRAAdapter` weights through `BypassInjectionManager`. Each scheduled adapter stores its own rank and absolute strength schedules. A lightweight `PREDICT_NOISE` wrapper publishes only the current sampler step index through thread-local state; this allows serial dynamic LoRA nodes to remain independent.

## Multi-stage sampling

A model may be sampled more than once in one graph. The plugin therefore stores all scheduled adapters in one model-patcher registry, aggregates adapters by target module, and keeps one stable bypass injection group. Before replacing that group it ejects the active group, preventing `original_forward` from pointing at a previous bypass wrapper.

## Public API

The node is model-only and exposes exactly four required inputs: `model`, `lora_name`, `rank_schedule`, and `strength_schedule`. Strength values are absolute values in `0..1`; the old static `strength_model` and `strength_clip` inputs are intentionally removed.

## Broadcast axes

- Linear outputs use `[..., rank]`, so the mask is shaped with rank on the last axis.
- Conv outputs use `[batch, rank, spatial...]`, so the mask is shaped with rank on axis 1.

## Native fallback

DoRA/reshape metadata, non-standard adapter families, and tuple/sliced mappings are kept on ComfyUI's native static patch path and produce warnings. This avoids silently changing their math while preserving load compatibility.
