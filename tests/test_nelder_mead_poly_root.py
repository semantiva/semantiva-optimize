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


def test_nelder_mead_poly_root():
    """Test Nelder-Mead gradient-free optimization finds polynomial roots."""
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
                "strategy": "nelder-mead",
                "x0": [0.1],
                "bounds": None,  # Nelder-Mead doesn't require bounds
                "termination": {"max_evals": 500, "ftol_abs": 1e-12, "xtol_abs": 1e-12},
                "model_name": "poly_residual",
                "model_params": {"coeffs": [1.0, 0.0, -2.0]},  # x^2 - 2
            },
        }
    ]

    # Execute pipeline
    pipeline = Pipeline(nodes)
    result = pipeline.process(Payload(data=NoDataType(), context=ContextType()))

    # Verify results
    best = result.context.get_value("optimizer.best_candidate")
    v = best["x"][0]
    assert abs(v - math.sqrt(2.0)) < 1e-4 or abs(v + math.sqrt(2.0)) < 1e-4
