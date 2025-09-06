from __future__ import annotations
from typing import Sequence


class LocalConvex:
    def initialize(self, *, x0, bounds, termination, model, controller, constraints, params, seed=None):
        self._model = model
        self._bounds = bounds
        self._constraints = constraints
        self._term = termination
        self._params = params or {}
        return {"iter": 0, "best": {"x": list(x0), "value": float("inf"), "feasible": True}}

    def ask(self, state):
        return state["best"]["x"]

    def tell(self, state, x, value, feasible, viol):
        state["iter"] += 1
        try:
            import scipy.optimize as opt
        except Exception:
            return state

        method = "L-BFGS-B"
        cons = ()
        if self._constraints and (self._constraints.ineq or self._constraints.eq):
            method = "SLSQP"
            cons_list = []
            for g in self._constraints.ineq:
                cons_list.append({"type": "ineq", "fun": (lambda z, g=g: -g(z))})
            for h in self._constraints.eq:
                cons_list.append({"type": "eq",   "fun": (lambda z, h=h:  h(z))})
            cons = tuple(cons_list)

        res = opt.minimize(
            fun=lambda z: float(self._model.objective(z)),
            x0=x,
            jac=(lambda z: self._model.gradient(z)) if hasattr(self._model, "gradient") and self._model.gradient(x) is not None else None,
            bounds=self._bounds,
            constraints=cons,
            method=method,
            options={"maxiter": self._term.max_evals, "ftol": self._term.ftol_abs},
        )
        state["best"] = {"x": [float(v) for v in res.x], "value": float(res.fun), "feasible": True}
        return state

    def should_stop(self, state):
        return state["iter"] >= 1

    def termination_summary(self, state):
        return {"reason": "completed", "metrics": {"iter": state["iter"]}, "budget": {"max_evals": self._term.max_evals}}
