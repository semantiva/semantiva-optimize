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

"""
Nelder-Mead optimization strategy using scipy.

This module implements a gradient-free optimization strategy using the
Nelder-Mead simplex algorithm for unconstrained optimization problems.
"""

from __future__ import annotations


class NelderMead:
    """Nelder-Mead simplex optimization strategy."""

    def __init__(self):
        """Initialize strategy with default parameters."""
        self._model = None
        self._term = None

    def initialize(
        self,
        *,
        x0,
        bounds,  # pylint: disable=unused-argument
        termination,
        model,
        controller,  # pylint: disable=unused-argument
        constraints,  # pylint: disable=unused-argument
        params,  # pylint: disable=unused-argument
        seed=None,  # pylint: disable=unused-argument
    ):
        """
        Initialize optimization state.

        Args:
            x0: Initial point
            bounds: Variable bounds (unused for Nelder-Mead)
            termination: Termination criteria
            model: Objective function model
            controller: Controller interface (unused)
            constraints: Optimization constraints (unused)
            params: Strategy parameters (unused)
            seed: Random seed (unused)

        Returns:
            Initial optimization state
        """
        self._model = model
        self._term = termination
        return {
            "iter": 0,
            "best": {"x": list(x0), "value": float("inf"), "feasible": True},
        }

    def ask(self, state):
        """Request next candidate point."""
        return state["best"]["x"]

    def tell(self, state, x, value, feasible, viol):  # pylint: disable=unused-argument
        """
        Provide objective function feedback and run optimizer.

        Args:
            state: Current optimization state
            x: Candidate point
            value: Objective value (unused, scipy computes directly)
            feasible: Feasibility flag (unused)
            viol: Constraint violation (unused)

        Returns:
            Updated optimization state
        """
        import scipy.optimize as opt  # pylint: disable=import-outside-toplevel

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
        """Check if optimization should terminate."""
        return state["iter"] >= 1

    def termination_summary(self, state):
        """Provide termination summary."""
        return {
            "reason": "completed",
            "metrics": {"iter": state["iter"]},
            "budget": {"max_evals": self._term.max_evals},
        }
