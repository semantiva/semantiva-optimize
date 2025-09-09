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

"""
Comprehensive tests demonstrating improved pipeline-based testing patterns.

These tests serve as examples of best practices for testing semantiva-optimize
components using realistic pipeline execution rather than internal APIs.
"""

import math
import pytest
import yaml
from semantiva import Pipeline, Payload
from semantiva.context_processors.context_types import ContextType
from semantiva.data_types import NoDataType
from semantiva.registry.plugin_registry import load_extensions


def test_yaml_based_optimization_complete():
    """Test optimization using complete YAML configuration (most realistic usage)."""
    try:
        import scipy  # noqa
    except ImportError:
        pytest.skip("SciPy not installed")

    # Complete YAML configuration as users would write it
    yaml_config = """
    extensions: ["semantiva_optimize"]
    
    pipeline:
      nodes:
        - processor: OptimizerContextProcessor
          parameters:
            strategy: "local"
            x0: [0.5] 
            bounds: [[-10, 10]]
            model_name: "poly_residual"
            model_params:
              coeffs: [1.0, 0.0, -2.0]  # x^2 - 2
            termination:
              max_evals: 200
              ftol_abs: 1.0e-12
              xtol_abs: 1.0e-12
            log_every: 10
    """

    config = yaml.safe_load(yaml_config)

    # Load extensions and execute pipeline
    load_extensions(config.get("extensions", []))
    nodes = config["pipeline"]["nodes"]
    pipeline = Pipeline(nodes)
    result = pipeline.process(Payload(data=NoDataType(), context=ContextType()))

    # Verify optimization results
    best = result.context.get_value("optimizer.best_candidate")
    v = best["x"][0]
    assert abs(v - math.sqrt(2.0)) < 1e-4 or abs(v + math.sqrt(2.0)) < 1e-4

    # Verify strategy information
    strategy = result.context.get_value("optimizer.strategy")
    assert strategy == "LocalConvex"

    # Verify history is captured
    history = result.context.get_value("optimizer.history")
    assert len(history) > 0
    assert all("x" in step and "value" in step for step in history)


def test_comprehensive_multi_start_optimization():
    """Test comprehensive multi-start optimization with full validation."""
    try:
        import scipy  # noqa
    except ImportError:
        pytest.skip("SciPy not installed")

    # Load extension
    load_extensions(["semantiva_optimize"])

    # Multi-start configuration
    nodes = [
        {
            "processor": "OptimizerContextProcessor",
            "parameters": {
                "strategy": "local",
                "x0": [0.0],  # Will be overridden by multi_start
                "multi_start": [[-5.0], [0.0], [5.0], [10.0]],  # Four starting points
                "bounds": [[-10.0, 10.0]],
                "termination": {"max_evals": 100, "ftol_abs": 1e-12},
                "model_name": "parabola",
                "model_params": {"x_star": 3.0},  # (x-3)^2
                "log_every": 5,
            },
        }
    ]

    # Execute pipeline
    pipeline = Pipeline(nodes)
    result = pipeline.process(Payload(data=NoDataType(), context=ContextType()))

    # Verify multi-start behavior
    runs = result.context.get_value("optimizer.runs")
    assert len(runs) == 4  # Four runs for four start points

    # Each run should have basic structure (not nested best_candidate)
    for i, run in enumerate(runs):
        assert "x" in run
        assert ("value" in run) or ("f" in run)
        if "f" in run:
            run_value = run["f"]
        else:
            run_value = run["value"]
        assert isinstance(run_value, float)

    # Global best should be at x=3
    best = result.context.get_value("optimizer.best_candidate")
    assert math.isclose(best["x"][0], 3.0, rel_tol=1e-4, abs_tol=1e-4)
    assert best["value"] < 1e-10


def test_nelder_mead_gradient_free_comprehensive():
    """Test Nelder-Mead strategy with comprehensive validation."""
    try:
        import scipy  # noqa
    except ImportError:
        pytest.skip("SciPy not installed")

    # Load extension
    load_extensions(["semantiva_optimize"])

    # Gradient-free optimization configuration
    nodes = [
        {
            "processor": "OptimizerContextProcessor",
            "parameters": {
                "strategy": "nelder-mead",
                "x0": [0.1],
                "bounds": None,  # Nelder-Mead doesn't require bounds
                "termination": {"max_evals": 1000, "ftol_abs": 1e-8, "xtol_abs": 1e-8},
                "model_name": "poly_residual",
                "model_params": {"coeffs": [1.0, 0.0, -2.0]},  # x^2 - 2
                "log_every": 50,
            },
        }
    ]

    # Execute pipeline
    pipeline = Pipeline(nodes)
    result = pipeline.process(Payload(data=NoDataType(), context=ContextType()))

    # Verify results
    best = result.context.get_value("optimizer.best_candidate")
    v = best["x"][0]
    assert abs(v - math.sqrt(2.0)) < 1e-3 or abs(v + math.sqrt(2.0)) < 1e-3

    # Verify strategy
    strategy = result.context.get_value("optimizer.strategy")
    assert strategy == "NelderMead"

    # Verify convergence
    history = result.context.get_value("optimizer.history")
    assert len(history) >= 1  # At least one iteration
    # Verify history has proper structure
    for entry in history:
        assert "value" in entry
        assert "iter" in entry


@pytest.mark.parametrize(
    "model_type,model_params,expected_min",
    [
        ("parabola", {"x_star": 2.0}, 2.0),
        ("parabola", {"x_star": -1.5}, -1.5),
        ("parabola", {"x_star": 0.0}, 0.0),
    ],
)
def test_parametrized_model_optimization(model_type, model_params, expected_min):
    """Test optimization with different model parameters (parametrized testing)."""
    try:
        import scipy  # noqa
    except ImportError:
        pytest.skip("SciPy not installed")

    # Load extension
    load_extensions(["semantiva_optimize"])

    # Parametrized configuration
    nodes = [
        {
            "processor": "OptimizerContextProcessor",
            "parameters": {
                "strategy": "local",
                "x0": [0.0],
                "bounds": [[-10.0, 10.0]],
                "termination": {"max_evals": 100, "ftol_abs": 1e-12},
                "model_name": model_type,
                "model_params": model_params,
            },
        }
    ]

    # Execute pipeline
    pipeline = Pipeline(nodes)
    result = pipeline.process(Payload(data=NoDataType(), context=ContextType()))

    # Verify convergence to expected minimum
    best = result.context.get_value("optimizer.best_candidate")
    assert math.isclose(best["x"][0], expected_min, rel_tol=1e-6, abs_tol=1e-6)
    assert best["value"] < 1e-10


def test_termination_criteria_validation():
    """Test different termination criteria work correctly."""
    try:
        import scipy  # noqa
    except ImportError:
        pytest.skip("SciPy not installed")

    # Load extension
    load_extensions(["semantiva_optimize"])

    # Test with max_evals termination
    nodes = [
        {
            "processor": "OptimizerContextProcessor",
            "parameters": {
                "strategy": "local",
                "x0": [5.0],  # Far from minimum
                "bounds": [[-10.0, 10.0]],
                "termination": {"max_evals": 5},  # Very limited evaluations
                "model_name": "parabola",
                "model_params": {"x_star": 0.0},
            },
        }
    ]

    # Execute pipeline
    pipeline = Pipeline(nodes)
    result = pipeline.process(Payload(data=NoDataType(), context=ContextType()))

    # Verify limited evaluations
    history = result.context.get_value("optimizer.history")
    assert len(history) <= 5  # Should not exceed max_evals

    # Best result exists even with early termination
    best = result.context.get_value("optimizer.best_candidate")
    assert "x" in best and "value" in best
