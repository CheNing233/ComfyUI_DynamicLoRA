# ComfyUI Dynamic LoRA

面向 ComfyUI 原生 LoRA 的离散采样步调度节点，支持：

- 按采样 step 动态启用 LoRA rank；
- 按采样 step 精确控制 LoRA strength；
- 多个动态 LoRA 串行连接；
- 同一个模型经过多个 KSampler 阶段；
- 生成波形形式的 rank/strength schedule 字符串；
- 按 step 展示 schedule 的实际离散取值。

作者：xChenNing

## 节点

### 1. Load LoRA (Dynamic Rank + Strength)

这是唯一的动态 LoRA 加载节点，模型专用，输入为：

```text
model
lora_name
rank_schedule
strength_schedule
```

没有额外的 `strength_model` 或 `strength_clip` 输入。

#### rank_schedule

按总 rank 的比例启用前缀 rank：

```text
1,0.5,1,0
```

例如 LoRA 总 rank 为 64：

```text
1.0 → 64 个 rank
0.5 → 32 个 rank
0.0 → 关闭 LoRA
```

rank 数量使用“四舍五入，`.5` 向上”的规则计算；例如总 rank 为 3、比例为 0.5 时激活 2 个 rank。

#### strength_schedule

绝对的每步 LoRA strength：

```text
1,0.5,1,0
```

表示：

```text
第 1 步：1.0
第 2 步：0.5
第 3 步：1.0
第 4 步：0.0
```

如果采样步数多于字符串长度，最后一个值会持续使用；如果采样步数少于字符串长度，多余值会被丢弃。

只控制动态 strength 时：

```text
rank_schedule = 1
strength_schedule = 1,0.5,1,0
```

### 2. LoRA Schedule Waveform

生成离散的 schedule 字符串，可直接连接到：

```text
rank_schedule
strength_schedule
```

支持波形：

```text
constant
linear_up
linear_down
cosine
triangle
square
pulse
```

主要参数：

```text
steps
waveform
start_value
end_value
min_value
max_value
phase
cycles
duty_cycle
pulse_start
pulse_end
invert
decimals
```

因为输出要直接连接到 LoRA rank/strength schedule，所有波形数值都限制在 `[0,1]`。

`linear_up` / `linear_down` 使用 `start_value` 和 `end_value`；周期型 `cosine`、`triangle`、`square`、`pulse` 使用 `min_value` 和 `max_value`。周期型波形采用半开区间采样，最后一个 step 不重复第一个 phase；例如 square 的 4 步、2 周期输出为 `1,0,1,0`。

波形输出严格包含 `steps` 个值，不做额外平滑或插值。

常用例子：

#### 交替 rank

```text
steps = 30
waveform = square
cycles = 15
```

输出类似：

```text
1,0,1,0,1,0,...
```

#### 线性关闭

```text
steps = 30
waveform = linear_down
```

#### 中间脉冲

```text
steps = 30
waveform = pulse
pulse_start = 0.3
pulse_end = 0.7
```

### 3. LoRA Schedule Preview

输入 schedule 字符串后，逐 step 展示实际值：

```text
steps: 4
step 0001: 1
step 0002: 0.5
step 0003: 1
step 0004: 0
```

该预览是离散 step 展示，不绘制连续平滑曲线。

如果字符串解析失败，会输出中文错误信息，例如：

```text
解析失败：step 2 is not a number: 'abc'
```

## Anima 多阶段采样兼容

插件使用一个稳定的 step wrapper 和一个稳定的 bypass injection group。多个动态 LoRA 会按目标模块聚合，避免同一个 Linear 被重复包裹。

因此支持类似：

```text
KSampler
  → Ultimate SD Upscale
      → 第二个 KSampler
```

两个采样阶段可以复用同一个模型。

## 原生兼容范围

动态 bypass 路径支持标准 ComfyUI `LoRAAdapter`，包括：

- Anima 风格 `lora_up.weight` / `lora_down.weight`；
- 普通原生 LoRA；
- 带 `mid` 权重的 LoCon。

以下类型会回退到 ComfyUI 原生静态 patch，并输出 warning：

- DoRA / reshape metadata；
- LoHa；
- LoKr；
- OFT / BOFT；
- GLoRA；
- tuple/sliced mapping。

这些回退类型不会获得逐 step rank/strength 动态控制，但会尽量保持原生数学行为。

## 测试

在当前 ComfyUI Python 环境运行：

```powershell
F:\SDComfyUI\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

纯 schedule 测试：

```powershell
python -m unittest tests/test_schedule.py tests/test_waveform.py
```

## 安装位置

当前工作站安装目录：

```text
F:\SDComfyUI\custom_nodes\ComfyUI-DynamicLoraRank
```

修改插件后请重启 ComfyUI，避免旧的模型 hook 保留在 Python 进程中。
