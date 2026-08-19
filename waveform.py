"""Discrete schedule generators and a single-chart step preview renderer."""

from __future__ import annotations

import math
from typing import Sequence


OSCILLATION_WAVEFORMS = ("square", "cosine")
MONOTONIC_CURVES = ("linear", "cosine", "exponential", "logarithmic")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _validate_steps(steps: int) -> int:
    steps = int(steps)
    if steps < 1 or steps > 8192:
        raise ValueError("steps must be between 1 and 8192")
    return steps


def _validate_range(min_value: float, max_value: float) -> tuple[float, float]:
    min_value = float(min_value)
    max_value = float(max_value)
    if not math.isfinite(min_value) or not math.isfinite(max_value):
        raise ValueError("min_value and max_value must be finite numbers")
    if not math.isfinite(min_value) or not math.isfinite(max_value):
        raise ValueError("min_value and max_value must be finite numbers")
    if min_value > max_value:
        raise ValueError("min_value cannot be greater than max_value")
    return min_value, max_value


def _step_window(steps: int, start_step: int, end_step: int) -> tuple[int, int]:
    """Return an inclusive zero-based step window from 1-based user inputs."""
    start = max(1, min(steps, int(start_step)))
    end = steps if int(end_step) == 0 else max(1, min(steps, int(end_step)))
    if start > end:
        raise ValueError("start_step cannot be greater than end_step")
    return start - 1, end - 1


def _format_value(value: float, decimals: int) -> str:
    text = f"{float(value):.{max(0, min(12, int(decimals)))}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def format_schedule(values: Sequence[float], decimals: int = 6) -> str:
    return ",".join(_format_value(value, decimals) for value in values)


def generate_oscillation_schedule(
    steps: int,
    waveform: str,
    x_step_offset: float = 0.0,
    y_offset: float = 0.5,
    amplitude: float = 0.5,
    cycles: float = 1.0,
    min_value: float = 0.0,
    max_value: float = 1.0,
    start_step: int = 1,
    end_step: int = 0,
    decimals: int = 6,
) -> tuple[float, ...]:
    """Generate square/cosine oscillation values in an inclusive step window.

    Outside the active window the output is zero, which makes the result useful
    as a direct LoRA rank/strength schedule.
    """
    steps = _validate_steps(steps)
    waveform = str(waveform).strip().lower()
    if waveform not in OSCILLATION_WAVEFORMS:
        raise ValueError(f"unknown oscillation waveform: {waveform!r}")
    min_value, max_value = _validate_range(min_value, max_value)
    x_step_offset = float(x_step_offset)
    y_offset = float(y_offset)
    amplitude = float(amplitude)
    cycles = float(cycles)
    if not all(math.isfinite(value) for value in (x_step_offset, y_offset, amplitude, cycles)):
        raise ValueError("x_step_offset, y_offset, amplitude and cycles must be finite numbers")
    if cycles < 0.0:
        raise ValueError("cycles cannot be negative")
    left, right = _step_window(steps, start_step, end_step)
    window_length = max(1, right - left + 1)
    values = []
    for index in range(steps):
        if index < left or index > right:
            values.append(0.0)
            continue
        # Oscillation windows use half-open step sampling so square waves
        # remain truly alternating at the final active step.
        position = ((index - left) + x_step_offset) / float(window_length)
        phase = 2.0 * math.pi * cycles * position
        carrier = 1.0 if (phase % (2.0 * math.pi)) < math.pi else -1.0
        if waveform == "cosine":
            carrier = math.cos(phase)
        values.append(round(_clamp(y_offset + amplitude * carrier, min_value, max_value), decimals))
    return tuple(values)


def _monotonic_progress(curve: str, position: float) -> float:
    position = _clamp(position)
    if curve == "linear":
        return position
    if curve == "cosine":
        return 0.5 - 0.5 * math.cos(math.pi * position)
    if curve == "exponential":
        curvature = 4.0
        return math.expm1(curvature * position) / math.expm1(curvature)
    if curve == "logarithmic":
        curvature = 4.0
        return math.log1p(curvature * position) / math.log1p(curvature)
    raise ValueError(f"unknown monotonic curve: {curve!r}")


def generate_monotonic_schedule(
    steps: int,
    curve: str,
    left_value: float = 0.0,
    right_value: float = 1.0,
    start_step: int = 1,
    end_step: int = 0,
    decimals: int = 6,
) -> tuple[float, ...]:
    """Generate a monotonic schedule with held left/right limits."""
    steps = _validate_steps(steps)
    curve = str(curve).strip().lower()
    if curve not in MONOTONIC_CURVES:
        raise ValueError(f"unknown monotonic curve: {curve!r}")
    left_value = float(left_value)
    right_value = float(right_value)
    if not math.isfinite(left_value) or not math.isfinite(right_value):
        raise ValueError("left_value and right_value must be finite numbers")
    start, end = _step_window(steps, start_step, end_step)
    span = max(1, end - start)
    values = []
    for index in range(steps):
        if index <= start:
            values.append(round(left_value, decimals))
        elif index >= end:
            values.append(round(right_value, decimals))
        else:
            progress = _monotonic_progress(curve, (index - start) / float(span))
            values.append(round(left_value + progress * (right_value - left_value), decimals))
    return tuple(values)



def flow_shift_schedule(values: Sequence[float], flow_shift: float = 3.0, invert: bool = True, decimals: int = 6) -> tuple[float, ...]:
    """Remap values with the Anima/Flow discrete-flow shift toward high noise.

    ``progress=0`` is the first/high-noise step and ``progress=1`` is the
    final/low-noise step. With ``invert=True`` (the recommended Anima
    compensation), a shift greater than one samples a later source progress,
    pulling values toward the low-noise side. ``invert=False`` applies the
    opposite high-noise pull.
    """
    values = tuple(float(value) for value in values)
    if not values:
        raise ValueError("schedule string is empty")
    flow_shift = float(flow_shift)
    if not math.isfinite(flow_shift) or flow_shift <= 0.0:
        raise ValueError("flow_shift must be a positive finite number")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("schedule values must be finite numbers")
    if len(values) == 1 or flow_shift == 1.0:
        return tuple(round(value, decimals) for value in values)

    output = []
    last_index = len(values) - 1
    for index in range(len(values)):
        progress = index / float(last_index)
        if invert:
            source_progress = flow_shift * progress / (1.0 + (flow_shift - 1.0) * progress)
        else:
            noise = 1.0 - progress
            shifted_noise = flow_shift * noise / (1.0 + (flow_shift - 1.0) * noise)
            source_progress = 1.0 - shifted_noise
        source_progress = _clamp(source_progress)
        source_position = source_progress * last_index
        left = int(math.floor(source_position))
        right = min(last_index, left + 1)
        fraction = source_position - left
        value = values[left] + (values[right] - values[left]) * fraction
        output.append(round(value, decimals))
    return tuple(output)


def parse_numeric_schedule(
    value: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> tuple[float, ...]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("schedule string is empty")
    tokens = [token.strip() for token in value.split(",")]
    if any(token == "" for token in tokens):
        raise ValueError("empty value between commas")
    values = []
    for index, token in enumerate(tokens, start=1):
        try:
            number = float(token)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"step {index} is not a number: {token!r}") from exc
        if not math.isfinite(number):
            raise ValueError(f"step {index} is not finite: {token!r}")
        if (min_value is not None and number < min_value) or (max_value is not None and number > max_value):
            raise ValueError(f"step {index} is outside the allowed range [{min_value}, {max_value}]: {token!r}")
        values.append(number)
    return tuple(values)


def _catmull_rom(values: Sequence[float], samples_per_segment: int = 12) -> list[float]:
    if len(values) <= 1:
        return list(values)
    result = []
    for index in range(len(values) - 1):
        p0 = values[max(0, index - 1)]
        p1 = values[index]
        p2 = values[index + 1]
        p3 = values[min(len(values) - 1, index + 2)]
        for sample in range(samples_per_segment):
            t = sample / float(samples_per_segment)
            value = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t)
            result.append(value)
    result.append(float(values[-1]))
    return result


def render_schedule_preview(values: Sequence[float], decimals: int = 6):
    """Render one chart with smooth curve and discrete points on the same axes."""
    from PIL import Image, ImageDraw, ImageFont

    values = tuple(float(value) for value in values)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("schedule values must be finite numbers")
    font = ImageFont.load_default()
    y_min = min(0.0, min(values))
    y_max = max(1.0, max(values))
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0
    width = 1200
    graph_height = 520
    margin_left, margin_right, margin_top = 82, 28, 42
    table_top = graph_height + 70
    columns = min(8, max(1, len(values)))
    rows = (len(values) + columns - 1) // columns
    height = table_top + rows * 18 + 36
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    graph_left = margin_left
    graph_right = width - margin_right
    graph_top = margin_top
    graph_bottom = graph_height
    draw.text((margin_left, 12), "XCN Schedule Preview", fill=(30, 30, 30), font=font)
    for tick in range(5):
        y_value = y_min + (y_max - y_min) * tick / 4.0
        y = graph_bottom - int((y_value - y_min) / (y_max - y_min) * (graph_bottom - graph_top))
        draw.line((graph_left, y, graph_right, y), fill=(225, 225, 225), width=1)
        draw.text((20, y - 5), _format_value(y_value, 2), fill=(80, 80, 80), font=font)
    draw.line((graph_left, graph_top, graph_left, graph_bottom), fill=(70, 70, 70), width=2)
    draw.line((graph_left, graph_bottom, graph_right, graph_bottom), fill=(70, 70, 70), width=2)

    def point(step_index: int, value: float):
        x = graph_left if len(values) == 1 else graph_left + (graph_right - graph_left) * step_index / (len(values) - 1)
        y = graph_bottom - (graph_bottom - graph_top) * ((value - y_min) / (y_max - y_min))
        return int(round(x)), int(round(y))

    smooth = _catmull_rom(values)
    smooth_points = []
    for i, value in enumerate(smooth):
        source_position = i / max(1, len(smooth) - 1)
        smooth_points.append((int(round(graph_left + (graph_right - graph_left) * source_position)), int(round(graph_bottom - (graph_bottom - graph_top) * ((value - y_min) / (y_max - y_min))))))
    if len(smooth_points) > 1:
        draw.line(smooth_points, fill=(40, 110, 220), width=3)

    discrete_points = [point(i, value) for i, value in enumerate(values)]
    if len(discrete_points) > 1:
        draw.line(discrete_points, fill=(230, 120, 30), width=1)
    for index, (x, y) in enumerate(discrete_points):
        draw.line((x, graph_bottom, x, y), fill=(245, 190, 120), width=1)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(230, 90, 20), outline=(120, 50, 10))

    draw.text((margin_left, graph_bottom + 16), "蓝线：平滑趋势    橙点：每个离散 step 的实际取值", fill=(60, 60, 60), font=font)
    cell_width = (width - margin_left - margin_right) // columns
    for index, value in enumerate(values):
        row, column = divmod(index, columns)
        x = margin_left + column * cell_width
        y = table_top + row * 18
        draw.text((x, y), f"step {index + 1:04d}: {_format_value(value, decimals)}", fill=(30, 30, 30), font=font)
    return image


def format_step_preview(values: Sequence[float], decimals: int = 6) -> str:
    lines = [f"steps: {len(values)}"]
    for index, value in enumerate(values, start=1):
        lines.append(f"step {index:04d}: {_format_value(value, decimals)}")
    return "\n".join(lines)
