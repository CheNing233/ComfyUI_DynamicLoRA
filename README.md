# ComfyUI Dynamic LoRA

作者：xChenNing

面向 ComfyUI 原生 LoRA 与 Anima 的离散采样步调度插件。所有公开节点统一使用 `XCN_` 前缀。

## 节点

### XCN_DynamicLoraLoader

唯一的动态 LoRA 加载节点：

```text
model
lora_name
rank_schedule
strength_schedule
```

`rank_schedule` 控制 LoRA 激活的 rank 比例；`strength_schedule` 是绝对的每步 LoRA strength。两者都使用英文逗号分隔，缺少的后续 step 使用最后一个值，多余值丢弃。

```text
rank_schedule = 1,0.5,1,0
strength_schedule = 1,0.8,1,0
```

### XCN_OscillationSchedule

用于生成方波或余弦震荡 schedule。

输入：

```text
steps
waveform: square / cosine
x_step_offset
y_offset
amplitude
cycles
min_value
max_value
start_step
end_step
```

`x_step_offset` 是 X 轴的 step 偏移；`y_offset` 是中心线；`amplitude` 是振幅；`cycles` 是生效区间内的周期数；`min_value` / `max_value` 是最终裁剪范围。`start_step` 和 `end_step` 使用从 1 开始的 step 编号，`end_step=0` 表示最后一步。生效区间外输出 0。

例如：

```text
steps = 8
waveform = square
x_step_offset = 0
y_offset = 0.5
amplitude = 0.5
cycles = 4
```

输出：

```text
1,0,1,0,1,0,1,0
```

### XCN_MonotonicSchedule

用于生成单调变化 schedule：

```text
linear
cosine
exponential
logarithmic
```

输入：

```text
steps
curve
left_value
right_value
start_step
end_step
```

生效区间之前保持 `left_value`，之后保持 `right_value`。

### XCN_FlowShiftSchedule

把已有 schedule 按 Anima/Flow 的 `shift` 关系重新取样，让取值向高噪声 steps 侧延伸。

输入：

```text
schedule
flow_shift
invert
```

Anima 默认使用：

```text
flow_shift = 3.0
invert = true
```

Anima 原生 shift=3 已经让高噪声区域拥有更多采样密度，因此本节点默认用 `invert=true` 把 schedule 取值向低噪声侧补偿。

例如原始 schedule：

```text
0,0.5,1
```

使用 `flow_shift=3, invert=true` 后，中间值会向低噪声侧移动；使用 `invert=false` 则是向高噪声侧移动。`flow_shift=1` 表示不改变 schedule。

内部关系为：

```text
invert=true:
source_progress = shift * progress / (1 + (shift - 1) * progress)

invert=false:
noise = 1 - progress
shifted_noise = shift * noise / (1 + (shift - 1) * noise)
source_progress = 1 - shifted_noise
```

它可以放在任意 schedule 发生器和 LoRA Loader 之间：

```text
XCN_OscillationSchedule / XCN_MonotonicSchedule
  → XCN_FlowShiftSchedule
      → XCN_DynamicLoraLoader
```

### XCN_SchedulePreview

输入 schedule 字符串，输出一张图。图中只有一组 X/Y 坐标：

- 蓝色平滑趋势线；
- 橙色离散 step 折线与圆点；
- 每个离散点的竖向辅助线；
- 底部列出每个 step 的具体取值。

平滑线只用于观察趋势，实际控制仍然使用橙色离散点。

如果字符串解析失败或数值超出 `[0,1]`，不生成正常曲线，而是显示中文失败文案。

## 值域与 step 语义

所有用于 LoRA rank/strength 的值都限制在：

```text
0 <= value <= 1
```

周期型震荡采用离散 step 的半开区间采样，例如 4 步、2 周期的方波为：

```text
1,0,1,0
```

## Anima 多阶段采样

插件使用单一稳定的 step wrapper 和单一稳定 injection group。多个动态 LoRA 会按目标模块聚合，支持：

```text
KSampler
  → Ultimate SD Upscale
      → 第二个 KSampler
```

两个采样阶段可以复用同一个模型。

## 原生兼容

动态 bypass 支持标准 ComfyUI `LoRAAdapter`、普通原生 LoRA、Anima LoRA，以及带 `mid` 的 LoCon。DoRA/reshape、LoHa、LoKr、OFT/BOFT、GLoRA、tuple/sliced mapping 会回退到 ComfyUI 原生静态 patch，并输出 warning。

## 测试

在当前 ComfyUI Python 环境运行：

```powershell
F:\SDComfyUI\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

训练器仓库中的纯 schedule 测试：

```powershell
python -m unittest tests/test_dynamic_lora_rank_schedule.py
```

## 安装

```text
F:\SDComfyUI\custom_nodes\ComfyUI-DynamicLoraRank
```

更新插件后请重启 ComfyUI，避免旧的模型 forward hook 留在进程中。
