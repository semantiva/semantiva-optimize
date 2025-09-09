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

from __future__ import annotations

from typing import Callable, Optional, Sequence, Any, cast

import scipy.optimize as opt

from ._scipy_shim import ObjectiveRecorder


class NelderMead:
    """Gradient-free Nelder-Mead optimization."""

    def __init__(self) -> None:
        self._on_iter: Optional[Callable[[int, Sequence[float], float], None]] = None

    def set_progress_broadcaster(
        self, on_iter: Callable[[int, Sequence[float], float], None]
    ) -> None:
        self._on_iter = on_iter

    def tell(
        self,
        x0: Sequence[float],
        model,
        bounds,  # unused
        constraints,  # unused
        termination,
        **kwargs,
    ):
        rec = ObjectiveRecorder(lambda z: model.objective(z))

        iter_idx = {"k": 0}

        def _callback(xk):
            f_val = rec.last_f if rec.last_x is xk else float(model.objective(xk))
            if self._on_iter is not None:
                self._on_iter(iter_idx["k"], xk, f_val)
            iter_idx["k"] += 1

        options = {}
        if termination and getattr(termination, "max_evals", None) is not None:
            options["maxiter"] = termination.max_evals
        if termination and getattr(termination, "ftol_abs", None) is not None:
            options["fatol"] = termination.ftol_abs
        if termination and getattr(termination, "xtol_abs", None) is not None:
            options["xatol"] = termination.xtol_abs

        # Use a casted reference to minimize to avoid mypy overload resolution errors
        _opt_minimize = cast(Any, opt.minimize)
        res = _opt_minimize(
            fun=cast(Any, rec.fun),
            x0=x0,
            method="Nelder-Mead",
            callback=cast(Any, _callback),
            options=options,
        )
        return res
