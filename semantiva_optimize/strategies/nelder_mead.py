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


class NelderMead:
    def initialize(
        self,
        *,
        x0,
        bounds,
        termination,
        model,
        controller,
        constraints,
        params,
        seed=None,
    ):
        self._model = model
        self._term = termination
        return {
            "iter": 0,
            "best": {"x": list(x0), "value": float("inf"), "feasible": True},
        }

    def ask(self, state):
        return state["best"]["x"]

    def tell(self, state, x, value, feasible, viol):
        import scipy.optimize as opt

        res = opt.minimize(
            fun=lambda z: float(self._model.objective(z)),
            x0=x,
            method="Nelder-Mead",
            options={
                "maxiter": self._term.max_evals,
                "fatol": self._term.ftol_abs,
                "xatol": self._term.xtol_abs,
            },
        )
        state["iter"] += 1
        state["best"] = {
            "x": [float(v) for v in res.x],
            "value": float(res.fun),
            "feasible": True,
        }
        return state

    def should_stop(self, state):
        return state["iter"] >= 1

    def termination_summary(self, state):
        return {
            "reason": "completed",
            "metrics": {"iter": state["iter"]},
            "budget": {"max_evals": self._term.max_evals},
        }
