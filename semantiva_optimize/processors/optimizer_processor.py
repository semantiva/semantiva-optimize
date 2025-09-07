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
from typing import Sequence

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
        # NEW knobs:
        log_every: int | None = None,
        model_name: str | None = None,
        model_params: dict | None = None,
        decimals: int = 6,
    ) -> None:
        """
        Main optimization processing logic.

        Handles strategy resolution, parameter processing, and multi-start optimization
        with comprehensive logging and context updates.
        """
        # Resolve strategy and model from strings/parameters
        strategy, model = self._resolve_strategy_and_model(
            strategy, strategy_params, model, model_name, model_params
        )

        # Process termination criteria
        termination = self._process_termination(termination)

        # Set up context
        self._setup_optimization_context(strategy, bounds, termination, strategy_params)

        # Run optimization
        starts = multi_start or [x0]
        runs, global_best, final_term = self._run_multi_start_optimization(
            starts,
            strategy,
            bounds,
            termination,
            model,
            controller,
            constraints,
            strategy_params,
            seed,
            log_every,
            decimals,
        )

        # Update final context
        self._finalize_optimization_context(runs, global_best, final_term)

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

    def _run_multi_start_optimization(
        self,
        starts,
        strategy,
        bounds,
        termination,
        model,
        controller,
        constraints,
        strategy_params,
        seed,
        log_every,
        decimals,
    ):
        """Run optimization from multiple starting points."""
        runs, global_best = [], None
        log_every = max(0, int(log_every or 0))
        final_term = None

        for start_x in starts:
            best, term = self._run_single_optimization(
                start_x,
                strategy,
                bounds,
                termination,
                model,
                controller,
                constraints,
                strategy_params,
                seed,
                log_every,
                decimals,
            )
            runs.append(best)
            if global_best is None:
                global_best = best
            elif best and best["value"] < global_best["value"]:
                global_best = best
            final_term = term  # Keep the last termination info

        return runs, global_best, final_term

    def _run_single_optimization(
        self,
        start_x,
        strategy,
        bounds,
        termination,
        model,
        controller,
        constraints,
        strategy_params,
        seed,
        log_every,
        decimals,
    ):
        """Run optimization from a single starting point."""
        if self.logger:
            self.logger.debug(
                f"[optimize] start strategy={strategy.__class__.__name__} "
                f"bounds={bounds} x0={_fmt_vec(start_x, decimals)}"
            )

        state = strategy.initialize(
            x0=start_x,
            bounds=bounds,
            termination=termination,
            model=model,
            controller=controller,
            constraints=constraints,
            params=strategy_params,
            seed=seed,
        )

        best = self._optimization_loop(
            strategy, state, model, controller, constraints, log_every, decimals
        )

        term = strategy.termination_summary(state)
        if self.logger:
            self.logger.debug(
                f"[optimize] done reason={term.get('reason')} "
                f"best_x={_fmt_vec(best['x'], decimals)} best_f={best['value']:.{decimals}g}"
            )

        return best, term

    def _optimization_loop(
        self, strategy, state, model, controller, constraints, log_every, decimals
    ):
        """Main optimization iteration loop."""
        neval = 0
        best = None

        while not strategy.should_stop(state):
            x = strategy.ask(state)
            value, feasible, viol = self._evaluate_candidate(
                x, model, controller, constraints
            )

            step = self._create_step_record(
                state, x, value, feasible, viol, neval, strategy
            )
            self._update_history(state, step)

            if self.logger and log_every and (neval % log_every == 0):
                self._log_iteration(step, decimals)

            state = strategy.tell(state, x, value, feasible, viol)
            neval += 1
            best = self._update_best_candidate(state, best, x, value, feasible)

        return best or {
            "x": state.get("x0", []),
            "value": float("nan"),
            "feasible": False,
            "meta": {},
        }

    def _evaluate_candidate(self, x, model, controller, constraints):
        """Evaluate a candidate solution."""
        if controller:
            feasible = controller.safe(x)
            obs = controller.apply(x)
            value = model.objective(x) if model else float(obs)
        else:
            feasible = True
            value = model.objective(x) if model else float("nan")

        viol = 0.0
        if constraints:
            for g in constraints.ineq:
                viol = max(viol, 0.0, g(x))  # Fixed nested max call
            for h in constraints.eq:
                viol = max(viol, abs(h(x)))
            feasible = feasible and (viol <= 1e-12)

        return value, feasible, viol

    def _create_step_record(self, state, x, value, feasible, viol, neval, strategy):
        """Create a step record for the optimization history."""
        return {
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
                "context": {"bounds": state.get("bounds")},
                "trace": {"iter": state.get("iter", 0)},
            },
        }

    def _update_history(self, state, step):
        """Update optimization history in state and context."""
        hist = state.setdefault("__history__", [])
        hist.append(step)
        self._notify_context_update(self.HISTORY_KEY, hist)

    def _log_iteration(self, step, decimals):
        """Log iteration details."""
        self.logger.debug(
            f"[optimize] iter={step['iter']} x={_fmt_vec(step['x'], decimals)} "
            f"f={step['value']:.{decimals}g} feas={step['feasible']} "
            f"viol={step['violations']:.{decimals}g}"
        )

    def _update_best_candidate(self, state, current_best, x, value, feasible):
        """Update the best candidate found so far."""
        cand = state.get("best", {"x": x, "value": value, "feasible": feasible})
        if (current_best is None) or (cand["value"] < current_best["value"]):
            return {
                "x": list(cand["x"]),  # Fixed unsubscriptable issue
                "value": float(cand["value"]),
                "feasible": bool(cand["feasible"]),
                "meta": {},
            }
        return current_best

    def _finalize_optimization_context(self, runs, global_best, final_term):
        """Finalize optimization context with results."""
        self._notify_context_update("optimizer.runs", runs)
        self._notify_context_update(self.BEST_KEY, global_best)
        self._notify_context_update(self.TERMINATION_KEY, final_term)

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
