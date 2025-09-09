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

from typing import Callable, Optional, Sequence


class ObjectiveRecorder:
    """Caches last (x, f) so SciPy callback doesn't re-evaluate objective."""

    def __init__(
        self,
        fun: Callable[[Sequence[float]], float],
        jac: Optional[Callable[[Sequence[float]], Sequence[float]]] = None,
    ) -> None:
        self._fun = fun
        self._jac = jac
        self.last_x: Optional[Sequence[float]] = None
        self.last_f: Optional[float] = None
        self.neval = 0

    def fun(self, x: Sequence[float]) -> float:
        self.last_x = x
        self.last_f = float(self._fun(x))
        self.neval += 1
        return self.last_f

    def jac(self, x: Sequence[float]):
        if self._jac is None:
            return None
        return self._jac(x)
