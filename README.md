# ComfyUI Dynamic LoRA

作者：xChenNing

一个面向 ComfyUI 原生 LoRA 与 Anima 的离散采样步调度插件。所有公开节点统一使用 `XCN_` 前缀。

## 节点

### XCN_DynamicLoraLoader

唯一的动态 LoRA 加载节点，输入：

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

专门生成方波或余弦震荡 schedule。

核心参数：

```text
steps
waveform: square / cosine
x_cycles
y_offset
amplitude
min_value / max_value
start_step / end_step
```

`start_step` 与 `end_step` 使用从 1 开始的 step 编号；`end_step=0` 表示最后一步。生效区间之外输出 0。输出始终是离散 step 值。

例如：

```text
steps = 8
waveform = square
x_cycles = 4
y_offset = 0.5
amplitude = 0.5
```

输出：

```text
1,0,1,0,1,0,1,0
```

### XCN_MonotonicSchedule

专门生成单调曲线：

```text
linear
cosine
exponential
logarithmic
```

核心参数：

```text
steps
curve
left_value
right_value
start_step
end_step
```

生效区间之前保持 `left_value`，之后保持 `right_value`。

### XCN_SchedulePreview

输入 schedule 字符串，输出一张图：

- 平滑趋势线；
- 所有离散 step 的实际取值点；
- 离散点与趋势线使用同一组 X/Y 坐标；
- 底部列出每个 step 的具体数值。

例如：

```text
step 0001: 1
step 0002: 0.5
step 0003: 1
step 0004: 0
```

如果解析失败或数值超出 `[0,1]`，预览节点不显示正常曲线，而显示中文错误文案。

## 波形与值域

所有用于 LoRA rank/strength 的输出值限制在：

```text
0 <= value <= 1
```

周期型震荡采用离散 step 的半开区间采样，避免最后一个 step 重复第一 phase。

## Anima 多阶段采样

插件使用单一稳定的 step wrapper 和单一稳定 injection group。多个动态 LoRA 会按目标模块聚合，支持同一个模型经过多个 KSampler 阶段，例如：

```text
KSampler
  → Ultimate SD Upscale
      → 第二个 KSampler
```

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
