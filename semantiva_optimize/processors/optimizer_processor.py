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
Context processor for optimization workflows.

This module provides the OptimizerContextProcessor that integrates optimization
strategies into Semantiva's context processing pipeline, supporting multi-start
optimization, constraint handling, and epistemic transparency.
"""

from __future__ import annotations
from typing import Sequence, Optional, Dict, Any

from semantiva.context_processors.context_processors import ContextProcessor

from semantiva_optimize.termination import Termination
from semantiva_optimize.adapters.model_adapter import ModelAdapter
from semantiva_optimize.adapters.controller_adapter import ControllerAdapter
from semantiva_optimize.constraints import Constraints
from semantiva_optimize.factory import make_strategy
from semantiva_optimize.examples.models import make_model as _make_example_model


def _fmt_vec(x, decimals: int = 6) -> str:
    """Format vector for logging with specified decimal places."""
    try:
        return "[" + ", ".join(f"{float(v):.{decimals}g}" for v in x) + "]"
    except Exception:  # pylint: disable=broad-exception-caught
        return str(x)


class OptimizerContextProcessor(ContextProcessor):
    """
    Context processor for optimization workflows.

    Integrates optimization strategies into Semantiva's context processing pipeline
    with support for multi-start optimization, constraints, and epistemic transparency.
    """

    STRATEGY_KEY = "optimizer.strategy"
    PARAMS_KEY = "optimizer.params"
    HISTORY_KEY = "optimizer.history"
    BEST_KEY = "optimizer.best_candidate"
    TERMINATION_KEY = "optimizer.termination"

    def _process_logic(
        self,
        *,
        strategy,
        x0: Sequence[float],
        bounds: list[tuple[float, float]] | None = None,
        termination: Termination | None = None,
        model: ModelAdapter | None = None,
        controller: ControllerAdapter | None = None,
        constraints: Constraints | None = None,
        strategy_params: dict | None = None,
        seed: int | None = None,
        multi_start: list[Sequence[float]] | None = None,
        # Optional objective spec (YAML friendly): { class: "module.Class", kwargs: { ... } }
        objective: dict | None = None,
        # NEW knobs:
        log_every: int | None = None,
        model_name: str | None = None,
        model_params: dict | None = None,
        decimals: int = 6,
        progress=None,
        progress_update_every: int = 1,
        progress_throttle_s: float = 0.0,
    ) -> None:
        """
        Main optimization processing logic.

        Handles strategy resolution, parameter processing, and multi-start optimization
        with comprehensive logging and context updates.
        """
        # Resolve strategy and model from strings/parameters
        if model is None and objective:
            try:
                cls_path = (
                    objective.get("class") if isinstance(objective, dict) else None
                )
                kw = objective.get("kwargs", {}) if isinstance(objective, dict) else {}
                if cls_path:
                    import importlib

                    module_name, class_name = cls_path.rsplit(".", 1)
                    mod = importlib.import_module(module_name)
                    cls = getattr(mod, class_name)
                    model = cls(**(kw or {}))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                raise ValueError(f"Failed to instantiate objective from spec: {exc}")

        strategy, model = self._resolve_strategy_and_model(
            strategy, strategy_params, model, model_name, model_params
        )

        termination = self._process_termination(termination)
        self._setup_optimization_context(strategy, bounds, termination, strategy_params)

        from semantiva_optimize.progress.base import StartEvent, StepEvent, EndEvent

        history: list[dict] = []
        self._notify_context_update(self.HISTORY_KEY, history)

        def _append_history(record: dict):
            history.append(record)
            self._notify_context_update(self.HISTORY_KEY, history)

        def _emit_progress(event_ctor, payload: dict, append_to_history: bool = False):
            if append_to_history and event_ctor is StepEvent:
                rec = dict(payload)
                rec["value"] = rec.pop("f")
                _append_history(rec)
            e = event_ctor(**payload)
            for ob in observers:
                try:
                    if event_ctor is StartEvent:
                        ob.on_start(e)
                    elif event_ctor is StepEvent:
                        ob.on_step(e)
                        if payload.get("is_best"):
                            ob.on_best(e)
                    else:
                        ob.on_end(e)
                except Exception:
                    pass
            if event_ctor is EndEvent:
                for ob in observers:
                    try:
                        ob.close()
                    except Exception:
                        pass

        progress_update_every = int(progress_update_every or 1)
        progress_throttle_s = float(progress_throttle_s or 0.0)
        if progress is None:
            observers = []
        elif isinstance(progress, (list, tuple)):
            observers = list(progress)
        else:
            observers = [progress]

        import time

        def _run_one_start(run_id: Optional[int], start_x) -> Dict[str, Any]:
            _emit_progress(
                StartEvent,
                {
                    "total_runs": total_runs,
                    "run_id": run_id,
                    "meta": {"strategy": strategy_name},
                },
            )

            best: Dict[str, Any] = {"x": list(start_x), "f": float("inf")}
            last_emit = {"t": 0.0}

            def _feas_and_viol(x):
                if constraints is None:
                    return True, 0.0
                max_viol = 0.0
                feas = True
                for g in constraints.ineq or []:
                    v = max(0.0, float(g(x)))
                    max_viol = max(max_viol, v)
                    feas = feas and (v == 0.0)
                for h in constraints.eq or []:
                    v = abs(float(h(x)))
                    max_viol = max(max_viol, v)
                    feas = feas and (v == 0.0)
                return feas, max_viol

            def _is_best(f: float) -> bool:
                return f < best["f"]

            def _on_iter(iter_idx: int, xk, fk: float) -> None:
                if iter_idx % progress_update_every != 0:
                    return
                now = time.time()
                if (
                    progress_throttle_s > 0
                    and (now - last_emit["t"]) < progress_throttle_s
                ):
                    return
                last_emit["t"] = now
                feasible, viol = _feas_and_viol(xk)
                is_new_best = _is_best(fk)
                if is_new_best:
                    best["x"], best["f"] = list(xk), float(fk)
                payload = {
                    "iter": iter_idx,
                    "x": list(xk),
                    "f": float(fk),
                    "feasible": bool(feasible),
                    "violation": float(viol),
                    "run_id": run_id,
                    "is_best": bool(is_new_best),
                    "meta": {"strategy": strategy_name, "source": "scipy"},
                }
                _emit_progress(StepEvent, payload, append_to_history=True)

            if hasattr(strategy, "set_progress_broadcaster"):
                strategy.set_progress_broadcaster(_on_iter)

            res = strategy.tell(
                x0=start_x,
                model=model,
                bounds=bounds,
                constraints=constraints,
                termination=termination,
            )

            feasible, viol = _feas_and_viol(res.x)
            if res.fun < best["f"]:
                best["x"], best["f"] = list(res.x), float(res.fun)

            _emit_progress(
                EndEvent,
                {
                    "reason": getattr(res, "message", "completed"),
                    "best_x": best["x"],
                    "best_f": best["f"],
                    "run_id": run_id,
                    "meta": {"strategy": strategy_name},
                },
            )
            return best

        starts = multi_start or [x0]
        strategy_name = strategy.__class__.__name__
        total_runs = len(starts)
        runs: list[dict] = []
        for rid, start in enumerate(starts):
            runs.append(_run_one_start(rid if total_runs > 1 else None, start))

        global_best: Optional[Dict[str, Any]] = None
        for r in runs:
            if global_best is None:
                global_best = r
            else:
                # now safe to index global_best
                if r["f"] < global_best["f"]:
                    global_best = r

        self._notify_context_update("optimizer.runs", runs)
        # mypy cannot prove runs is non-empty; assert to narrow type for indexing
        assert global_best is not None
        self._notify_context_update(
            self.BEST_KEY,
            {"x": global_best["x"], "value": global_best["f"], "feasible": True},
        )
        self._notify_context_update(
            self.TERMINATION_KEY,
            {"reason": "completed"},
        )

    def _resolve_strategy_and_model(
        self, strategy, strategy_params, model, model_name, model_params
    ):
        """Resolve strategy and model from various input types."""
        # YAML-friendly resolution
        if isinstance(strategy, str):
            strategy = make_strategy(strategy, **(strategy_params or {}))
        if (model is None) and model_name:
            model = _make_example_model(model_name, **(model_params or {}))
        return strategy, model

    def _process_termination(self, termination):
        """Process and normalize termination criteria."""
        # Convert termination dict to Termination object if needed
        if isinstance(termination, dict):
            # Ensure float values are properly converted
            float_fields = ["ftol_abs", "ftol_rel", "xtol_abs"]
            for field in float_fields:
                if field in termination:
                    termination[field] = float(termination[field])
            termination = Termination(**termination)
        return termination or Termination()

    def _setup_optimization_context(
        self, strategy, bounds, termination, strategy_params
    ):
        """Set up initial optimization context variables."""
        strategy_params = strategy_params or {}
        self._notify_context_update(self.STRATEGY_KEY, strategy.__class__.__name__)

        # Handle termination properly - convert to dict if it's an object
        termination_dict = (
            termination.__dict__ if hasattr(termination, "__dict__") else termination
        )
        self._notify_context_update(
            self.PARAMS_KEY,
            {
                "bounds": bounds,
                "termination": termination_dict,
                "strategy": strategy_params,
            },
        )
        self._notify_context_update(self.HISTORY_KEY, [])

    # required abstract methods
    def get_required_keys(self):
        """Get list of required context keys."""
        return []

    def get_created_keys(self):
        """Get list of context keys created by this processor."""
        return [
            self.STRATEGY_KEY,
            self.PARAMS_KEY,
            self.HISTORY_KEY,
            self.BEST_KEY,
            self.TERMINATION_KEY,
            "optimizer.runs",
        ]

    def get_suppressed_keys(self):
        """Get list of context keys suppressed by this processor."""
        return []
