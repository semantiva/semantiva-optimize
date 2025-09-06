from __future__ import annotations
from typing import Optional, Sequence
from semantiva.logger import Logger
from semantiva.context_processors.context_processors import ContextProcessor
from semantiva.context_processors.context_observer import _ContextObserver
from semantiva.context_processors.context_types import ContextType

from semantiva_optimize.termination import Termination
from semantiva_optimize.adapters.model_adapter import Model
from semantiva_optimize.adapters.controller_adapter import Controller
from semantiva_optimize.constraints import Constraints


class OptimizerContextProcessor(ContextProcessor):
    STRATEGY_KEY    = "optimizer.strategy"
    PARAMS_KEY      = "optimizer.params"
    HISTORY_KEY     = "optimizer.history"
    BEST_KEY        = "optimizer.best_candidate"
    TERMINATION_KEY = "optimizer.termination"

    def __init__(self, logger: Optional[Logger] = None):
        super().__init__(logger)
        self._ctx: ContextType | None = None
        self._observer: _ContextObserver | None = None

    # override operate_context to support observer and extra params
    def operate_context(self, *, context: ContextType, context_observer: _ContextObserver, **kwargs) -> ContextType:  # type: ignore[override]
        self._ctx = context
        self._observer = context_observer
        self._process_logic(**kwargs)
        return context

    # helper for notifying
    def _notify_context_update(self, key: str, value) -> None:
        if self._observer is not None and self._ctx is not None:
            self._observer.update_context(self._ctx, key, value)

    def _process_logic(
        self,
        *,
        strategy,
        x0: Sequence[float],
        bounds: list[tuple[float, float]] | None = None,
        termination: Termination | None = None,
        model: Model | None = None,
        controller: Controller | None = None,
        constraints: Constraints | None = None,
        strategy_params: dict | None = None,
        seed: int | None = None,
    ) -> None:
        termination = termination or Termination()
        strategy_params = strategy_params or {}
        self._notify_context_update(self.STRATEGY_KEY, strategy.__class__.__name__)
        self._notify_context_update(self.PARAMS_KEY, {
            "bounds": bounds, "termination": termination.__dict__, "strategy": strategy_params
        })
        self._notify_context_update(self.HISTORY_KEY, [])

        state = strategy.initialize(
            x0=x0, bounds=bounds, termination=termination,
            model=model, controller=controller, constraints=constraints,
            params=strategy_params, seed=seed
        )
        neval = 0
        best = None

        while not strategy.should_stop(state):
            x = strategy.ask(state)
            if controller:
                feasible = controller.safe(x)
                obs = controller.apply(x)
                value = model.objective(x) if model else float(obs)
            else:
                feasible = True
                value = model.objective(x) if model else float("nan")

            viol = 0.0
            if constraints:
                for g in constraints.ineq: viol = max(viol, max(0.0, g(x)))
                for h in constraints.eq:   viol = max(viol, abs(h(x)))
                feasible = feasible and (viol <= 1e-12)

            step = {
                "iter": state.get("iter", 0),
                "x": [float(v) for v in x],
                "value": float(value),
                "feasible": bool(feasible),
                "violations": float(viol),
                "step_info": {"neval": neval},
                "rng": state.get("rng"),
                "meu": {
                    "claim": {"value": float(value), "feasible": bool(feasible)},
                    "justification": {"rule": strategy.__class__.__name__},
                    "context": {"bounds": bounds},
                    "trace": {"iter": state.get("iter", 0)},
                },
            }
            history = state.setdefault("__history__", [])
            history.append(step)
            self._notify_context_update(self.HISTORY_KEY, history)

            state = strategy.tell(state, x, value, feasible, viol)
            neval += 1
            cand = state.get("best", {"x": x, "value": value, "feasible": feasible})
            if (best is None) or (cand["value"] < best["value"]):
                best = {"x": cand["x"], "value": float(cand["value"]), "feasible": bool(cand["feasible"]), "meta": {}}

        self._notify_context_update(self.BEST_KEY, best)
        self._notify_context_update(self.TERMINATION_KEY, strategy.termination_summary(state))

    # required abstract methods
    def get_required_keys(self):
        return []

    def get_created_keys(self):
        return [
            self.STRATEGY_KEY,
            self.PARAMS_KEY,
            self.HISTORY_KEY,
            self.BEST_KEY,
            self.TERMINATION_KEY,
        ]

    def get_suppressed_keys(self):
        return []
