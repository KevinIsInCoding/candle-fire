"""Shared result type for gate checks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Status = Literal["pass", "fail", "skip"]


@dataclass
class CheckResult:
    """Outcome of one gate check.

    A ``skip`` never fails the build — it means the check couldn't run (e.g. no
    gold labels yet, or runtime data absent in this suite). ``fail`` turns the
    gate red.
    """

    name: str
    suite: Literal["offline", "data"]
    status: Status
    summary: str                       # one-line human summary
    metrics: dict = field(default_factory=dict)   # measured values (for trend)
    thresholds: dict = field(default_factory=dict)  # gate thresholds applied
    details: list[str] = field(default_factory=list)  # per-item failures

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    @property
    def failed(self) -> bool:
        return self.status == "fail"

    @property
    def skipped(self) -> bool:
        return self.status == "skip"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "suite": self.suite,
            "status": self.status,
            "summary": self.summary,
            "metrics": self.metrics,
            "thresholds": self.thresholds,
            "details": self.details,
        }
