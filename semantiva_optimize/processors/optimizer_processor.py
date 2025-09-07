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
from typing import Optional, Sequence

from semantiva.logger import Logger
from semantiva.context_processors.context_processors import ContextProcessor

from semantiva_optimize.termination import Termination
from semantiva_optimize.adapters.model_adapter import ModelAdapter
from semantiva_optimize.adapters.controller_adapter import ControllerAdapter
from semantiva_optimize.constraints import Constraints
from semantiva_optimize.factory import make_strategy
from semantiva_optimize.examples.models import make_model as _make_example_model


def _fmt_vec(x, decimals: int = 6) -> str:
    try:
        return "[" + ", ".join(f"{float(v):.{decimals}g}" for v in x) + "]"
    except Exception:
        return str(x)


class OptimizerContextProcessor(ContextProcessor):
    STRATEGY_KEY = "optimizer.strategy"
    PARAMS_KEY = "optimizer.params"
    HISTORY_KEY = "optimizer.history"
    BEST_KEY = "optimizer.best_candidate"
    TERMINATION_KEY = "optimizer.termination"

    def __init__(self, logger: Optional[Logger] = None):
        super().__init__(logger)

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
        # YAML-friendly resolution
        if isinstance(strategy, str):
            strategy = make_strategy(strategy, **(strategy_params or {}))
        if (model is None) and model_name:
            model = _make_example_model(model_name, **(model_params or {}))

        # Convert termination dict to Termination object if needed
        if isinstance(termination, dict):
            # Ensure float values are properly converted
            if "ftol_abs" in termination:
                termination["ftol_abs"] = float(termination["ftol_abs"])
            if "ftol_rel" in termination:
                termination["ftol_rel"] = float(termination["ftol_rel"])
            if "xtol_abs" in termination:
                termination["xtol_abs"] = float(termination["xtol_abs"])
            termination = Termination(**termination)
        termination = termination or Termination()
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

        starts = multi_start or [x0]
        runs, global_best = [], None
        log_every = max(0, int(log_every or 0))

        for start_x in starts:
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
                    for g in constraints.ineq:
                        viol = max(viol, max(0.0, g(x)))
                    for h in constraints.eq:
                        viol = max(viol, abs(h(x)))
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
                hist = state.setdefault("__history__", [])
                hist.append(step)
                self._notify_context_update(self.HISTORY_KEY, hist)

                if self.logger and log_every and (neval % log_every == 0):
                    self.logger.debug(
                        f"[optimize] iter={step['iter']} x={_fmt_vec(step['x'], decimals)} "
                        f"f={step['value']:.{decimals}g} feas={step['feasible']} "
                        f"viol={step['violations']:.{decimals}g}"
                    )

                state = strategy.tell(state, x, value, feasible, viol)
                neval += 1
                cand = state.get("best", {"x": x, "value": value, "feasible": feasible})
                if (best is None) or (cand["value"] < best["value"]):
                    best = {
                        "x": cand["x"],
                        "value": float(cand["value"]),
                        "feasible": bool(cand["feasible"]),
                        "meta": {},
                    }

            if best is None:
                best = {
                    "x": start_x,
                    "value": float("nan"),
                    "feasible": False,
                    "meta": {},
                }
            runs.append(best)
            if (global_best is None) or (best["value"] < global_best["value"]):
                global_best = best

            term = strategy.termination_summary(state)
            if self.logger:
                self.logger.debug(
                    f"[optimize] done reason={term.get('reason')} "
                    f"best_x={_fmt_vec(best['x'], decimals)} best_f={best['value']:.{decimals}g}"
                )

        self._notify_context_update("optimizer.runs", runs)
        self._notify_context_update(self.BEST_KEY, global_best)
        self._notify_context_update(self.TERMINATION_KEY, term)

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
            "optimizer.runs",
        ]

    def get_suppressed_keys(self):
        return []
