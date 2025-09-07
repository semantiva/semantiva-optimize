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

from typing import Protocol, Sequence, Any


class ControllerAdapter(Protocol):
    def reset(self, seed: int | None = None) -> None: ...
    def apply(self, x: Sequence[float]) -> Any: ...
    def safe(self, x: Sequence[float]) -> bool: ...


class NullController:
    def reset(self, seed: int | None = None) -> None: ...

    def apply(self, x):
        return 0.0

    def safe(self, x):
        return True
