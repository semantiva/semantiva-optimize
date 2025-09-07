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
"""Parameter resolvers for ``opt.*`` strings."""
from __future__ import annotations

import re
from typing import Any

__all__ = ["optimize_param_resolver"]

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

_KV_RE = re.compile(r"\s*([A-Za-z_]\w*)\s*=\s*([^,]+)\s*")


def _parse_kv_list(s: str) -> dict[str, Any]:
    """Parse ``key=value`` comma-separated pairs into a dict."""
    out: dict[str, Any] = {}
    for match in _KV_RE.finditer(s):
        key, val = match.group(1), match.group(2)
        if val.lower() in {"true", "false"}:
            out[key] = val.lower() == "true"
            continue
        try:
            if "." in val or "e" in val.lower():
                out[key] = float(val)
            else:
                out[key] = int(val)
        except ValueError:
            out[key] = val
    return out


def optimize_param_resolver(value: Any) -> Any:
    """Resolve ``opt.*`` strings into class descriptors."""
    if not isinstance(value, str):
        return value
    if not value.startswith("opt."):
        return value

    body = value[4:]
    if body.startswith("strategy:"):
        alias = body.split(":", 1)[1].strip().lower()
        class_path = _STRATEGY_ALIASES.get(alias)
        if not class_path:
            raise ValueError(f"[opt.resolver] Unknown strategy alias: {alias!r}")
        return {"class": class_path, "kwargs": {}}

    if body.startswith("termination:"):
        kv = body.split(":", 1)[1]
        kwargs = _parse_kv_list(kv)
        return {
            "class": "semantiva_optimize.termination.Termination",
            "kwargs": kwargs,
        }

    if body.startswith("controller:"):
        sig = body.split(":", 1)[1].strip()
        match = re.match(r"^([A-Za-z_][\w\.]+)\((.*)\)$", sig)
        if not match:
            return {"class": sig, "kwargs": {}}
        class_path, kvs = match.group(1), match.group(2)
        kwargs = _parse_kv_list(kvs)
        return {"class": class_path, "kwargs": kwargs}

    return value
