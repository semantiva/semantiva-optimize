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
"""Config preprocessor for OptimizerContextProcessor.

This module implements a node-level config processor that converts friendly
YAML blocks into instantiable class descriptors. The processor only touches
nodes whose ``processor`` field refers to :class:`OptimizerContextProcessor`.
"""
from __future__ import annotations

from typing import Any, Dict

__all__ = ["optimize_config_preprocessor", "_resolve_controller_alias"]

# Accept both short and fully-qualified names for the processor
_OPT_PROCESSOR_NAMES = {
    "OptimizerContextProcessor",
    "semantiva_optimize.processors.optimizer_processor.OptimizerContextProcessor",
}

# Strategy alias table
_STRATEGY_ALIASES = {
    "local": "semantiva_optimize.strategies.local_convex.LocalConvex",
    "local-convex": "semantiva_optimize.strategies.local_convex.LocalConvex",
    "local_convex": "semantiva_optimize.strategies.local_convex.LocalConvex",
    "lbfgsb": "semantiva_optimize.strategies.local_convex.LocalConvex",
    "slsqp": "semantiva_optimize.strategies.local_convex.LocalConvex",
    "nelder": "semantiva_optimize.strategies.nelder_mead.NelderMead",
    "nelder-mead": "semantiva_optimize.strategies.nelder_mead.NelderMead",
    "nelder_mead": "semantiva_optimize.strategies.nelder_mead.NelderMead",
    "neldermead": "semantiva_optimize.strategies.nelder_mead.NelderMead",
}


def _resolve_controller_alias(alias: str) -> str:
    """Resolve a short controller alias to a Python class path."""
    aliases = {
        "imaging.sim": "semantiva_imaging.controllers.SimController",
    }
    return aliases.get(alias, alias)


def optimize_config_preprocessor(node: Dict[str, Any]) -> Dict[str, Any]:
    """Transform friendly YAML blocks into descriptors.

    Parameters
    ----------
    node:
        Pipeline node configuration.

    Returns
    -------
    dict
        Possibly mutated node dictionary with descriptors.
    """
    if node.get("processor") not in _OPT_PROCESSOR_NAMES:
        return node

    params = dict(node.get("parameters", {}))

    # Strategy: alias string -> descriptor
    strategy = params.get("strategy")
    if isinstance(strategy, str):
        key = strategy.strip().lower()
        class_path = _STRATEGY_ALIASES.get(key)
        if not class_path:
            raise ValueError(f"[opt] Unknown strategy alias: {strategy!r}")
        kwargs = dict(params.get("strategy_params") or {})
        params["strategy"] = {"class": class_path, "kwargs": kwargs}
        params.pop("strategy_params", None)

    # Termination: raw dict -> descriptor
    term = params.get("termination")
    if isinstance(term, dict):
        params["termination"] = {
            "class": "semantiva_optimize.termination.Termination",
            "kwargs": term,
        }

    # Controller alias form -> descriptor
    ctrl = params.get("controller")
    if isinstance(ctrl, dict) and "type" in ctrl and "params" in ctrl:
        class_path = _resolve_controller_alias(ctrl["type"])
        params["controller"] = {"class": class_path, "kwargs": dict(ctrl["params"])}

    # Constraints block -> descriptor (and extract bounds)
    cons = params.get("constraints")
    if isinstance(cons, dict) and any(k in cons for k in ("bounds", "ineq", "eq")):
        if "bounds" in cons and "bounds" not in params:
            params["bounds"] = cons["bounds"]
        params["constraints"] = {
            "class": "semantiva_optimize.descriptors.LinearConstraints",
            "kwargs": cons,
        }

    node["parameters"] = params
    return node
