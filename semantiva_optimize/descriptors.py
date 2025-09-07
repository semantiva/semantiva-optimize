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
"""Descriptor helpers for YAML-friendly configuration."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, cast

import numpy as np

from .constraints import Constraints

__all__ = ["LinearConstraints"]


def _as_vec(x: List[float]) -> np.ndarray:
    return np.asarray(x, dtype=float)


class LinearConstraints(Constraints):
    """Builder for linear constraints and bounds.

    Parameters are provided in a YAML-friendly structure and converted into
    callable inequality/equality functions using the ``g(x) <= 0`` convention.
    """

    def __init__(
        self,
        bounds: Optional[List[List[float]]] = None,
        ineq: Optional[List[Dict[str, Any]]] = None,
        eq: Optional[List[Dict[str, Any]]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._spec = {
            "bounds": bounds,
            "ineq": ineq or [],
            "eq": eq or [],
            "options": options or {},
        }

        # Constraints expected by the rest of the code take a sequence and
        # return either a numeric violation (float) or a list of floats.
        ineq_fns: List[Callable[[Sequence[float]], float | list[float]]] = []
        eq_fns: List[Callable[[Sequence[float]], float | list[float]]] = []

        local_ineq: List[Dict[str, Any]] = cast(List[Dict[str, Any]], ineq or [])
        local_eq: List[Dict[str, Any]] = cast(List[Dict[str, Any]], eq or [])

        for c in local_ineq:
            if not isinstance(c, dict):
                raise TypeError("inequality constraint spec must be a mapping")
            ctype = str(c.get("type", "linear")).lower()
            if ctype != "linear":
                raise ValueError(f"[opt.constraints] Unsupported ineq type: {ctype!r}")
            a_raw = c.get("a")
            if a_raw is None:
                raise KeyError("inequality constraint missing 'a' coefficient list")
            a = _as_vec(cast(List[float], a_raw))
            b = float(c.get("b", 0.0))

            def g(x, a=a, b=b):
                return [float(a @ _as_vec(x) - b)]

            ineq_fns.append(g)

        for c in local_eq:
            if not isinstance(c, dict):
                raise TypeError("equality constraint spec must be a mapping")
            ctype = str(c.get("type", "linear")).lower()
            if ctype != "linear":
                raise ValueError(f"[opt.constraints] Unsupported eq type: {ctype!r}")
            a_raw = c.get("a")
            if a_raw is None:
                raise KeyError("equality constraint missing 'a' coefficient list")
            a = _as_vec(cast(List[float], a_raw))
            b = float(c.get("b", 0.0))

            def h(x, a=a, b=b):
                return [float(a @ _as_vec(x) - b)]

            eq_fns.append(h)

        super().__init__(ineq=ineq_fns, eq=eq_fns)
        self.bounds = bounds
        self.spec = self._spec
