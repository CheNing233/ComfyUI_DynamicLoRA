"""Discrete waveform generation and schedule preview helpers."""

from __future__ import annotations

import math
from typing import Sequence


WAVEFORM_NAMES = (
    "constant",
    "linear_up",
    "linear_down",
    "cosine",
    "triangle",
    "square",
    "pulse",
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _phase_position(index: int, steps: int, cycles: float, phase: float) -> float:
    # Periodic waves use a half-open interval so the final step does not
    # duplicate the first phase (e.g. square(4, cycles=2) => 1,0,1,0).
    t = 0.0 if steps <= 1 else index / float(steps)
    return (t * cycles + phase) % 1.0


def generate_waveform(
    steps: int,
    waveform: str,
    start_value: float = 0.0,
    end_value: float = 1.0,
    min_value: float = 0.0,
    max_value: float = 1.0,
    phase: float = 0.0,
    cycles: float = 1.0,
    duty_cycle: float = 0.5,
    pulse_start: float = 0.25,
    pulse_end: float = 0.75,
    invert: bool = False,
    decimals: int = 6,
) -> tuple[float, ...]:
    """Generate exactly ``steps`` discrete values in the configured range."""
    if int(steps) < 1 or int(steps) > 8192:
        raise ValueError("steps must be between 1 and 8192")
    steps = int(steps)
    waveform = str(waveform).strip().lower()
    if waveform not in WAVEFORM_NAMES:
        raise ValueError(f"unknown waveform: {waveform!r}")
    values = [float(start_value), float(end_value), float(min_value), float(max_value)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("waveform values must be finite numbers")
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError("waveform values must be between 0 and 1")
    if min_value > max_value:
        raise ValueError("min_value cannot be greater than max_value")
    if not math.isfinite(float(phase)) or not math.isfinite(float(cycles)):
        raise ValueError("phase and cycles must be finite numbers")
    if cycles < 0.0:
        raise ValueError("cycles cannot be negative")
    if not 0.0 <= float(duty_cycle) <= 1.0:
        raise ValueError("duty_cycle must be between 0 and 1")
    if not 0.0 <= float(pulse_start) <= 1.0 or not 0.0 <= float(pulse_end) <= 1.0:
        raise ValueError("pulse_start and pulse_end must be between 0 and 1")
    if pulse_start > pulse_end:
        raise ValueError("pulse_start cannot be greater than pulse_end")
    decimals = max(0, min(12, int(decimals)))

    result: list[float] = []
    for index in range(steps):
        t = 0.0 if steps <= 1 else index / float(steps - 1)
        position = _phase_position(index, steps, cycles, phase)
        if waveform == "constant":
            normalized = 0.0
        elif waveform == "linear_up":
            normalized = _clamp(t + phase)
        elif waveform == "linear_down":
            normalized = 1.0 - _clamp(t + phase)
        elif waveform == "cosine":
            normalized = 0.5 - 0.5 * math.cos(2.0 * math.pi * position)
        elif waveform == "triangle":
            normalized = 1.0 - abs(2.0 * position - 1.0)
        elif waveform == "square":
            normalized = 1.0 if position < duty_cycle else 0.0
        elif waveform == "pulse":
            normalized = 1.0 if pulse_start <= position < pulse_end else 0.0
        else:  # pragma: no cover - guarded by waveform validation
            raise AssertionError(waveform)

        if invert:
            normalized = 1.0 - normalized
        if waveform in ("constant", "linear_up", "linear_down"):
            value = start_value if waveform == "constant" else start_value + normalized * (end_value - start_value)
        else:
            value = min_value + normalized * (max_value - min_value)
        value = _clamp(value, min_value, max_value)
        result.append(round(value, decimals))
    return tuple(result)


def format_schedule(values: Sequence[float], decimals: int = 6) -> str:
    decimals = max(0, min(12, int(decimals)))
    output = []
    for value in values:
        formatted = f"{float(value):.{decimals}f}".rstrip("0").rstrip(".")
        output.append(formatted if formatted else "0")
    return ",".join(output)


def parse_numeric_schedule(value: str, min_value: float | None = None, max_value: float | None = None) -> tuple[float, ...]:
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


def format_step_preview(values: Sequence[float], decimals: int = 6) -> str:
    decimals = max(0, min(12, int(decimals)))
    lines = [f"steps: {len(values)}"]
    for index, value in enumerate(values, start=1):
        formatted = f"{float(value):.{decimals}f}".rstrip("0").rstrip(".")
        lines.append(f"step {index:04d}: {formatted if formatted else '0'}")
    return "\n".join(lines)
