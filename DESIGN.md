# Design Notes

## Runtime model

The loader keeps ComfyUI's native key mapping and file conversion, but sends only standard `LoRAAdapter` weights through `BypassInjectionManager`. Each scheduled adapter stores its own rank and strength schedules. A lightweight `PREDICT_NOISE` wrapper publishes only the current sampler step index through thread-local state; this allows serial dynamic LoRA nodes to remain independent.

## Broadcast axes

- Linear outputs use `[..., rank]`, so the mask is shaped with rank on the last axis.
- Conv outputs use `[batch, rank, spatial...]`, so the mask is shaped with rank on axis 1.

## Native fallback

DoRA/reshape metadata, non-standard adapter families, and tuple/sliced mappings are kept on ComfyUI's native static patch path and produce warnings. This avoids silently changing their math while preserving load compatibility.
