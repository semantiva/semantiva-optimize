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
from semantiva import Pipeline, Payload
from semantiva.context_processors.context_types import ContextType
from semantiva.data_types import NoDataType
from semantiva.registry.plugin_registry import load_extensions


@pytest.mark.parametrize("x0", [[0.0], [10.0], [2.0]])
def test_local_convex_converges_to_3(x0):
    """Test local convex optimization converges to minimum at x=3 from different starting points."""
    try:
        import scipy  # noqa
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
                "x0": x0,
                "bounds": [[-100, 100]],
                "termination": {"max_evals": 100, "ftol_abs": 1e-12, "xtol_abs": 1e-12},
                "model_name": "parabola",
                "model_params": {"x_star": 3.0},  # (x-3)^2
            },
        }
    ]

    # Execute pipeline
    pipeline = Pipeline(nodes)
    result = pipeline.process(Payload(data=NoDataType(), context=ContextType()))

    # Verify results
    best = result.context.get_value("optimizer.best_candidate")
    assert abs(best["x"][0] - 3.0) < 1e-6
    assert best["value"] < 1e-12
