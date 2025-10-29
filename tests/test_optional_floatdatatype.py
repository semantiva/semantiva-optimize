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


def test_compat_floatdatatype_if_available():
    """Test compatibility with FloatDataType if available (optional dependency)."""
    try:
        from semantiva.examples.test_utils import FloatDataType
    except Exception:
        pytest.skip("FloatDataType not available")

    try:
        import scipy  # noqa
    except Exception:
        pytest.skip("SciPy not installed")

    # Load extension
    load_extensions(["semantiva-optimize"])

    # Extract float value from FloatDataType
    x0_value = FloatDataType(0.0).data

    # Create pipeline node configuration
    nodes = [
        {
            "processor": "OptimizerContextProcessor",
            "parameters": {
                "strategy": "local",
                "x0": [x0_value],
                "bounds": [[-10, 10]],
                "model_name": "parabola",
                "model_params": {"x_star": 1.0},  # (x-1)^2
                "termination": None,  # Uses default termination
            },
        }
    ]

    # Execute pipeline
    pipeline = Pipeline(nodes)
    result = pipeline.process(Payload(data=NoDataType(), context=ContextType()))

    # Verify results
    best = result.context.get_value("optimizer.best_candidate")
    assert abs(best["x"][0] - 1.0) < 1e-6
