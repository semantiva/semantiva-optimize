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

import pytest
from semantiva.registry import load_extensions
from semantiva.context_processors.context_observer import _ContextObserver
from semantiva.context_processors.context_types import ContextType


def test_safe_guard():
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
            return float(x[0] * x[0])

        def gradient(self, x):
            return [2.0 * x[0]]

    class GuardedController:
        def reset(self, seed=None):
            pass

        def apply(self, x):
            return 0.0

        def safe(self, x):
            return -1.0 <= x[0] <= 1.0

    p, ctx, obs = OptimizerContextProcessor(), ContextType(), _ContextObserver()
    p.operate_context(
        context=ctx,
        context_observer=obs,
        strategy=make_strategy("local"),
        x0=[5.0],
        bounds=[(-1.0, 1.0)],
        termination=Termination(max_evals=50),
        model=M(),
        controller=GuardedController(),
        constraints=None,
        strategy_params={},
    )
    best = ctx.get_value("optimizer.best_candidate")["x"][0]
    assert -1.0 <= best <= 1.0

