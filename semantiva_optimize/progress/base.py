# Copyright 2025 Semantiva authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
