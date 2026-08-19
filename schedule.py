"""Pure helpers for the Dynamic LoRA Rank ComfyUI node."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def parse_ratio_schedule(value: str | Iterable[float], field_name: str = "schedule") -> tuple[float, ...]:
    """Parse a comma-separated ratio schedule in the inclusive [0, 1] range."""
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",")]
        if not tokens or any(token == "" for token in tokens):
            raise ValueError(f"{field_name} must contain comma-separated numbers")
        raw_values = tokens
    else:
        raw_values = list(value)
        if not raw_values:
            raise ValueError(f"{field_name} must contain at least one value")

    result: list[float] = []
    for raw in raw_values:
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid {field_name} value: {raw!r}") from exc
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            raise ValueError(f"{field_name} values must be between 0 and 1: {raw!r}")
        result.append(number)
    return tuple(result)



def parse_strength_schedule(value: str | Iterable[float]) -> tuple[float, ...]:
    """Parse a strength schedule as finite floats without a [0, 1] limit."""
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",")]
        if not tokens or any(token == "" for token in tokens):
            raise ValueError("strength_schedule must contain comma-separated numbers")
        raw_values = tokens
    else:
        raw_values = list(value)
        if not raw_values:
            raise ValueError("strength_schedule must contain at least one value")

    result: list[float] = []
    for raw in raw_values:
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid strength_schedule value: {raw!r}") from exc
        if not math.isfinite(number):
            raise ValueError(f"strength_schedule values must be finite: {raw!r}")
        result.append(number)
    return tuple(result)

def parse_rank_schedule(value: str | Iterable[float]) -> tuple[float, ...]:
    """Backward-compatible alias for callers using the original helper name."""
    return parse_ratio_schedule(value, "rank_schedule")


def ratio_for_step(schedule: Sequence[float], step_index: int) -> float:
    """Use the last configured value for all steps beyond the schedule length."""
    if not schedule:
        raise ValueError("schedule cannot be empty")
    index = max(0, int(step_index))
    return float(schedule[min(index, len(schedule) - 1)])


def active_rank(total_rank: int, ratio: float) -> int:
    """Convert a rank ratio into a deterministic number of active leading dimensions."""
    if total_rank <= 0:
        return 0
    ratio = max(0.0, min(1.0, float(ratio)))
    # Round half up so rank=3 at 50% selects 2 dimensions, not 1.
    return max(0, min(int(total_rank), int(math.floor(total_rank * ratio + 0.5))))


def step_index_for_sigma(sigma: float, sample_sigmas: Sequence[float] | None) -> int:
    """Map a model sigma to a denoise-step index.

    ComfyUI exposes ``steps + 1`` sigmas, where the final entry is the terminal
    zero and is not a denoise model call. The index is therefore clamped to
    ``[0, len(sample_sigmas) - 2]``.
    """
    if sample_sigmas is None or len(sample_sigmas) == 0:
        return 0
    actual_steps = max(1, len(sample_sigmas) - 1)
    try:
        sigma_value = float(sigma)
        values = [float(item) for item in sample_sigmas[:actual_steps]]
        return max(0, min(actual_steps - 1, min(range(len(values)), key=lambda i: abs(values[i] - sigma_value))))
    except (TypeError, ValueError):
        return 0
