from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from paper_agent_v2.sandbox import FailureCategory, SandboxResult


@dataclass(frozen=True, slots=True)
class RepairAttempt:
    number: int
    target: str
    result: SandboxResult


class RestrictedRepairPatch(BaseModel):
    """The only mutation an LLM may request after sandbox failure."""

    target: Literal["ir", "custom_block", "glue"]
    node_id: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    params: dict[str, Any] | None = None
    inputs: list[str] | None = None
    custom_class_name: str | None = Field(default=None, pattern=r"^[A-Z][A-Za-z0-9_]*$")
    custom_source: str | None = None
    reason: str


def repair_priority(category: FailureCategory | None) -> tuple[str, ...]:
    if category in {FailureCategory.SHAPE, FailureCategory.DTYPE_DEVICE}:
        return ("ir", "custom_block", "glue")
    if category in {FailureCategory.SYNTAX, FailureCategory.IMPORT}:
        return ("custom_block", "glue", "ir")
    return ("ir", "custom_block", "glue")


def run_bounded_repairs(
    initial: SandboxResult,
    repair: Callable[[str, SandboxResult, int], SandboxResult],
    *,
    max_attempts: int = 3,
) -> tuple[SandboxResult, list[RepairAttempt]]:
    """Bounded orchestration only; callers may patch IR/custom/glue, never free-form projects."""
    current = initial
    attempts: list[RepairAttempt] = []
    for number in range(1, max_attempts + 1):
        if current.status == "passed":
            break
        targets = repair_priority(current.failure_category)
        target = targets[min(number - 1, len(targets) - 1)]
        current = repair(target, current, number)
        attempts.append(RepairAttempt(number=number, target=target, result=current))
    return current, attempts
