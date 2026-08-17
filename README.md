# ComfyUI Dynamic LoRA Rank + Strength

A standalone ComfyUI custom node that loads standard ComfyUI LoRAs and applies independent per-denoise-step rank and absolute strength schedules.

## Node

Only one public node is registered:

```text
Load LoRA (Dynamic Rank + Strength)
```

It is model-only, matching the Anima workflow usage.

## Inputs

```text
rank_schedule     = 1,0.5,1,0
strength_schedule = 1,0.5,1,0
```

`rank_schedule` masks the LoRA hidden rank. `strength_schedule` is the absolute LoRA strength at each denoise step. There is no separate `strength_model` input anymore.

For pure per-step strength control, use:

```text
rank_schedule     = 1
strength_schedule = 1,0.5,1,0
```

If the sampler has more steps than values, the final value repeats; extra values are ignored when the sampler has fewer steps. Values must be in `0..1`.

Each dynamic node keeps its own schedules, so serially connected dynamic LoRA nodes are independent.

## Multi-stage sampling compatibility

The implementation uses one stable `PREDICT_NOISE` wrapper and one stable bypass injection group per model patcher. Serial dynamic LoRAs are aggregated per target module, and the same model can be sampled again after a first KSampler stage (for example, an Ultimate SD Upscale subgraph) without stacking multiple bypass hooks on the same Linear layer.

## Compatibility

The dynamic bypass path supports standard ComfyUI `LoRAAdapter` weights, including Anima-style `lora_up.weight` / `lora_down.weight` checkpoints and LoCon `mid` weights. DoRA/reshape metadata, non-LoRA adapter families, and sliced/tuple mappings fall back to ComfyUI's native static patch path using the first strength value, with a warning; they are not dynamically scheduled.

CLIP is not part of this model-only node.

## Development

```powershell
python -m unittest discover -s tests -p 'test_*.py'
```

The installed copy used by this workstation is:

```text
F:\SDComfyUI\custom_nodes\ComfyUI-DynamicLoraRank
```
