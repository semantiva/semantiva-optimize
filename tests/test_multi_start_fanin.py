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


def test_multi_start_fan_in():
    try:
        import scipy  # noqa: F401
    except Exception:
        pytest.skip("SciPy not installed")

    from semantiva_optimize.processors.optimizer_processor import OptimizerContextProcessor
    from semantiva_optimize.factory import make_strategy
    from semantiva_optimize.termination import Termination

    load_extensions("semantiva_optimize.extension")

    class M:
        def objective(self, x):
            return float((x[0] - 2.0) ** 2)

        def gradient(self, x):
            return [2.0 * (x[0] - 2.0)]

    starts = [[-5.0], [0.0], [10.0]]

    p, ctx, obs = OptimizerContextProcessor(), ContextType(), _ContextObserver()
    p.operate_context(
        context=ctx,
        context_observer=obs,
        strategy=make_strategy("local"),
        x0=[0.0],
        multi_start=starts,
        bounds=[(-10.0, 10.0)],
        termination=Termination(max_evals=200, ftol_abs=1e-12, xtol_abs=1e-12),
        model=M(),
        controller=None,
        constraints=None,
        strategy_params={},
    )
    runs = ctx.get_value("optimizer.runs")
    assert len(runs) == 3
    v = ctx.get_value("optimizer.best_candidate")["x"][0]
    assert math.isclose(v, 2.0, rel_tol=1e-4, abs_tol=1e-4)

