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
from semantiva import Pipeline, Payload
from semantiva.context_processors.context_types import ContextType
from semantiva.data_types import NoDataType
from semantiva.registry.plugin_registry import load_extensions


def test_multi_start_fan_in():
    """Test multi-start optimization converges to global minimum from multiple starting points."""
    try:
        import scipy  # noqa: F401
    except Exception:
        pytest.skip("SciPy not installed")

    # Load extension
    load_extensions(["semantiva-optimize"])

    # Create pipeline node configuration
    nodes = [
        {
            "processor": "OptimizerContextProcessor",
            "parameters": {
                "strategy": "local",
                "x0": [0.0],  # Will be overridden by multi_start
                "multi_start": [[-5.0], [0.0], [10.0]],
                "bounds": [[-10.0, 10.0]],
                "termination": {"max_evals": 200, "ftol_abs": 1e-12, "xtol_abs": 1e-12},
                "model_name": "parabola",
                "model_params": {"x_star": 2.0},  # (x-2)^2
            },
        }
    ]

    # Execute pipeline
    pipeline = Pipeline(nodes)
    result = pipeline.process(Payload(data=NoDataType(), context=ContextType()))

    # Verify multi-start behavior
    runs = result.context.get_value("optimizer.runs")
    assert len(runs) == 3

    # Verify convergence to global minimum
    best = result.context.get_value("optimizer.best_candidate")
    v = best["x"][0]
    assert math.isclose(v, 2.0, rel_tol=1e-4, abs_tol=1e-4)
