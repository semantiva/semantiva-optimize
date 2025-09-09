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


class LocalConvex:
    """Local convex optimization using SciPy L-BFGS-B/SLSQP."""

    def __init__(self) -> None:
        self._on_iter: Optional[Callable[[int, Sequence[float], float], None]] = None

    def set_progress_broadcaster(
        self, on_iter: Callable[[int, Sequence[float], float], None]
    ) -> None:
        """Optional hook the processor may call; no-op if unused."""
        self._on_iter = on_iter

    def tell(
        self,
        x0: Sequence[float],
        model,
        bounds: Optional[Sequence[tuple[float, float]]],
        constraints,
        termination,
        **kwargs,
    ):
        jac = getattr(model, "gradient", None)
        if not callable(jac):
            jac = None
        rec = ObjectiveRecorder(lambda z: model.objective(z), jac)

        scipy_constraints = []
        if constraints is not None:
            for g in constraints.ineq or []:
                scipy_constraints.append(
                    {"type": "ineq", "fun": (lambda x, g=g: -float(g(x)))}
                )
            for h in constraints.eq or []:
                scipy_constraints.append(
                    {"type": "eq", "fun": (lambda x, h=h: float(h(x)))}
                )

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
            options["ftol"] = termination.ftol_abs

        method = "L-BFGS-B" if jac is not None else "SLSQP"

        # Use a casted reference to minimize to avoid mypy overload resolution errors
        _opt_minimize = cast(Any, opt.minimize)
        res = _opt_minimize(
            fun=cast(Any, rec.fun),
            jac=cast(Any, rec.jac) if jac is not None else None,
            x0=x0,
            bounds=bounds,
            constraints=scipy_constraints,
            method=method,
            callback=cast(Any, _callback),
            options=options,
        )
        return res
