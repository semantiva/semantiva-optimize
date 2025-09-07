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
Local convex optimization strategy using scipy optimizers.

This module implements a strategy that uses L-BFGS-B for bound-constrained
problems and SLSQP for general constrained optimization.
"""

from __future__ import annotations


class LocalConvex:
    """Local convex optimization using L-BFGS-B or SLSQP."""

    def __init__(self):
        """Initialize strategy with default parameters."""
        self._model = None
        self._bounds = None
        self._constraints = None
        self._term = None
        self._params = None

    def initialize(
        self,
        *,
        x0,
        bounds,
        termination,
        model,
        controller,  # pylint: disable=unused-argument
        constraints,
        params,
        seed=None,  # pylint: disable=unused-argument
    ):
        """
        Initialize optimization state.

        Args:
            x0: Initial point
            bounds: Variable bounds
            termination: Termination criteria
            model: Objective function model
            controller: Controller interface (unused)
            constraints: Optimization constraints
            params: Strategy parameters
            seed: Random seed (unused)

        Returns:
            Initial optimization state
        """
        self._model = model
        self._bounds = bounds
        self._constraints = constraints
        self._term = termination
        self._params = params or {}
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
        state["iter"] += 1
        try:
            import scipy.optimize as opt  # pylint: disable=import-outside-toplevel
        except Exception:  # pylint: disable=broad-exception-caught
            return state

        method, cons = self._setup_optimization_method()
        jac = self._setup_jacobian(x)

        res = opt.minimize(
            fun=lambda z: float(self._model.objective(z)),
            x0=x,
            jac=jac,
            bounds=self._bounds,
            constraints=cons,
            method=method,
            options={"maxiter": self._term.max_evals, "ftol": self._term.ftol_abs},
        )
        state["best"] = {
            "x": [float(v) for v in res.x],
            "value": float(res.fun),
            "feasible": True,
        }
        return state

    def _setup_optimization_method(self):
        """Set up optimization method and constraints."""
        method = "L-BFGS-B"
        cons = ()
        if self._constraints and (self._constraints.ineq or self._constraints.eq):
            method = "SLSQP"
            cons_list = []
            for g in self._constraints.ineq:
                cons_list.append({"type": "ineq", "fun": (lambda z, g=g: -g(z))})
            for h in self._constraints.eq:
                cons_list.append({"type": "eq", "fun": h})
            cons = tuple(cons_list)
        return method, cons

    def _setup_jacobian(self, x):
        """Set up jacobian function if available."""
        if hasattr(self._model, "gradient") and self._model.gradient(x) is not None:
            return self._model.gradient
        return None

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
