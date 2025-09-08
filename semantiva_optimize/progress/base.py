from dataclasses import dataclass
from typing import Sequence, Optional, Dict, Any, Protocol

__all__ = [
    "StartEvent",
    "StepEvent",
    "EndEvent",
    "ProgressObserver",
]


@dataclass
class StartEvent:
    total_runs: int
    run_id: Optional[int]
    meta: Dict[str, Any]


@dataclass
class StepEvent:
    iter: int
    x: Sequence[float]
    f: float
    feasible: bool
    violation: float
    run_id: Optional[int]
    is_best: bool
    meta: Dict[str, Any]


@dataclass
class EndEvent:
    reason: str
    best_x: Sequence[float]
    best_f: float
    run_id: Optional[int]
    meta: Dict[str, Any]


class ProgressObserver(Protocol):
    def on_start(self, e: StartEvent) -> None: ...
    def on_step(self, e: StepEvent) -> None: ...
    def on_best(self, e: StepEvent) -> None: ...
    def on_end(self, e: EndEvent) -> None: ...
    def close(self) -> None: ...
