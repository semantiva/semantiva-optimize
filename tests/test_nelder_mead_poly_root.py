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

import math
import pytest
from semantiva.registry import load_extensions
from semantiva.context_processors.context_observer import _ContextObserver
from semantiva.context_processors.context_types import ContextType


def test_nelder_mead_poly_root():
    try:
        import scipy  # noqa: F401
    except Exception:
        pytest.skip("SciPy not installed")

    from semantiva_optimize.processors.optimizer_processor import OptimizerContextProcessor
    from semantiva_optimize.factory import make_strategy
    from semantiva_optimize.termination import Termination

    load_extensions("semantiva_optimize.extension")

    class PolyModel:
        def objective(self, x):
            p = x[0] * x[0] - 2.0
            return float(p * p)

        def gradient(self, x):
            return None

    p = OptimizerContextProcessor()
    ctx, obs = ContextType(), _ContextObserver()
    p.operate_context(
        context=ctx,
        context_observer=obs,
        strategy=make_strategy("nelder-mead"),
        x0=[0.1],
        bounds=None,
        termination=Termination(max_evals=500, ftol_abs=1e-12, xtol_abs=1e-12),
        model=PolyModel(),
        controller=None,
        constraints=None,
        strategy_params={},
    )
    v = ctx.get_value("optimizer.best_candidate")["x"][0]
    assert abs(v - math.sqrt(2.0)) < 1e-4 or abs(v + math.sqrt(2.0)) < 1e-4

