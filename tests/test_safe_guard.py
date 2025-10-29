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
from semantiva import Pipeline
from semantiva.context_processors.context_types import ContextType
from semantiva.registry.plugin_registry import load_extensions


class GuardedController:
    """Test controller that ensures safety constraints."""

    def reset(self, seed=None):
        pass

    def apply(self, x):
        return 0.0

    def safe(self, x):
        return -1.0 <= x[0] <= 1.0


def test_safe_guard():
    """Test that safety guards work correctly with controllers."""
    try:
        import scipy  # noqa: F401
    except Exception:
        pytest.skip("SciPy not installed")

    # Load extension
    load_extensions(["semantiva-optimize"])

    # Create pipeline node configuration with controller
    # Note: Controllers need to be Python objects, not YAML-serializable
    # This test demonstrates a limitation of pure pipeline testing for complex objects
    nodes = [
        {
            "processor": "OptimizerContextProcessor",
            "parameters": {
                "strategy": "local",
                "x0": [5.0],  # Starting outside safe region
                "bounds": [[-1.0, 1.0]],
                "termination": {"max_evals": 50},
                "model_name": "parabola",
                "model_params": {"x_star": 0.0},  # x^2 (minimum at origin)
                # Note: controller parameter would need descriptor pattern for YAML
                # For now, we'll inject it post-creation for this test
            },
        }
    ]

    # Create pipeline and inject controller (demonstrates limitation)
    Pipeline(nodes)

    # For this advanced test, we need to modify the processor parameters
    # This shows where pipeline testing has limitations with complex objects
    from semantiva_optimize.processors.optimizer_processor import (
        OptimizerContextProcessor,
    )

    # Create processor with controller directly for this edge case
    processor = OptimizerContextProcessor()
    context = ContextType()
    from semantiva.context_processors.context_observer import _ContextObserver

    observer = _ContextObserver()

    # Use the processor directly for this complex controller test
    from semantiva_optimize.factory import make_strategy
    from semantiva_optimize.termination import Termination

    processor.operate_context(
        context=context,
        context_observer=observer,
        strategy=make_strategy("local"),
        x0=[5.0],
        bounds=[(-1.0, 1.0)],
        termination=Termination(max_evals=50),
        model_name="parabola",
        model_params={"x_star": 0.0},
        controller=GuardedController(),
        constraints=None,
        strategy_params={},
    )

    # Verify safety constraint is satisfied
    best = observer.observer_context.get_value("optimizer.best_candidate")["x"][0]
    assert -1.0 <= best <= 1.0
