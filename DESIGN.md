# Design Notes

## Public nodes

The public API has four prefixed nodes:

- `XCN_DynamicLoraLoader`
- `XCN_OscillationSchedule`
- `XCN_MonotonicSchedule`
- `XCN_SchedulePreview`

The two schedule generators intentionally have focused parameters instead of one generic parameter-heavy waveform node. Rank schedules remain ratios in `[0,1]`; strength schedules are arbitrary finite floats, including negative and above-one values.

## Oscillation generator

`x_step_offset` and `y_offset` define the horizontal/vertical offset; `cycles` controls the frequency and `amplitude` controls the oscillation size. `min_value`/`max_value` clip the result and `start_step`/`end_step` define an inclusive 1-based active window. Outside the window the output is zero.

## Monotonic generator

Linear, cosine, exponential, and logarithmic curves interpolate between `left_value` and `right_value`. Values before/after the active window hold the corresponding limit.

## Flow-shift remapper

`XCN_FlowShiftSchedule` applies the local Anima `ModelSamplingDiscreteFlow` mapping. With `shift=3.0` and `invert=True`, each output step samples a later normalized progress to compensate toward low noise; `invert=False` selects the opposite high-noise pull. `shift=1.0` is identity.

## Preview renderer

The previewer draws exactly one shared X/Y chart. A smooth Catmull-Rom trend line is overlaid with the exact discrete step polyline, point markers, stems, and a step/value table. Parsing or range errors return UI text instead of a normal chart.

## Multi-stage model lifecycle

Dynamic LoRA adapters are aggregated per target module into one stable bypass injection group. A stable step wrapper publishes the current sampler step. This prevents repeated KSampler stages from stacking bypass hooks on shared Anima Linear modules.
