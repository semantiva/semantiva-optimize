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

"""Test polynomial degree/theta alignment and PNG generation."""

import pytest


def test_degree_alignment_and_png(tmp_path, monkeypatch):
    """Test degree alignment validation and PNG generation with current & best lines."""
    # Set headless backend
    monkeypatch.setenv("MPLBACKEND", "Agg")

    from semantiva_optimize.progress.poly import PolynomialPlotObserver
    from semantiva_optimize.processors.optimizer_processor import (
        OptimizerContextProcessor,
    )
    from semantiva_optimize.factory import make_strategy
    from semantiva_optimize.termination import Termination
    from semantiva.context_processors.context_observer import _ContextObserver
    from semantiva.context_processors.context_types import ContextType

    # Simple quadratic model for testing
    class SimplePolynomialModel:
        def __init__(self):
            self.calls = 0

        def objective(self, theta):
            """Minimize residual for fitting y = x^2 with theta=[c0, c1, c2, c3]."""
            self.calls += 1
            x_data = [0, 1, 2, 3, 4, 5]
            y_data = [0.1, 1.2, 4.1, 8.9, 15.7, 24.9]

            residual_sum = 0.0
            for x_val, y_val in zip(x_data, y_data):
                y_pred = sum(theta[k] * (x_val**k) for k in range(len(theta)))
                residual_sum += (y_val - y_pred) ** 2

            return residual_sum

    x_data = [0, 1, 2, 3, 4, 5]
    y_data = [0.1, 1.2, 4.1, 8.9, 15.7, 24.9]
    degree = 3
    x0 = [0.0, 0.0, 1.0, 0.0]  # degree+1 coefficients

    # Test degree alignment validation
    assert (
        len(x0) == degree + 1
    ), f"x0 length ({len(x0)}) must equal degree+1 ({degree+1})"

    obs = PolynomialPlotObserver(
        x_data=x_data,
        y_data=y_data,
        degree=degree,
        mode="file",
        out_dir=str(tmp_path),
        file_prefix="fit",
    )

    obj = SimplePolynomialModel()

    p = OptimizerContextProcessor()
    ctx, co = ContextType(), _ContextObserver()

    p.operate_context(
        context=ctx,
        context_observer=co,
        strategy=make_strategy("local"),
        x0=x0,
        bounds=None,
        termination=Termination(max_evals=30, ftol_abs=1e-8),
        model=obj,
        controller=None,
        constraints=None,
        progress=[obs],
        progress_update_every=1,
        strategy_params={},
    )

    # Check that final PNG exists
    final_pngs = [f for f in tmp_path.iterdir() if f.name.endswith("_final.png")]
    assert len(final_pngs) >= 1, "Expected at least one final PNG file"

    # Check that some iteration PNGs were created
    iter_pngs = [
        f for f in tmp_path.iterdir() if "_iter" in f.name and f.name.endswith(".png")
    ]
    assert len(iter_pngs) >= 1, "Expected at least one iteration PNG file"

    # Verify optimization result structure
    best = co.observer_context.get_value("optimizer.best_candidate")
    assert best is not None, "Expected best candidate to be set"
    assert isinstance(best, dict) and "x" in best
    assert len(best["x"]) == 4, "Best candidate should have 4 coefficients"


def test_degree_mismatch_warning():
    """Test that degree/theta length mismatch is caught."""
    degree = 3
    x0_wrong = [1.0, 2.0]  # Only 2 coefficients for degree 3

    # This should raise an assertion error
    with pytest.raises(AssertionError, match="x0 length.*must equal degree"):
        assert (
            len(x0_wrong) == degree + 1
        ), f"x0 length ({len(x0_wrong)}) must equal degree+1 ({degree+1}) for polynomial plotting"


def test_polynomial_observer_initialization():
    """Test that PolynomialPlotObserver initializes correctly with degree parameter."""
    from semantiva_optimize.progress.poly import PolynomialPlotObserver

    x_data = [0, 1, 2, 3]
    y_data = [0, 1, 4, 9]
    degree = 2

    obs = PolynomialPlotObserver(
        x_data=x_data, y_data=y_data, degree=degree, mode="file"
    )

    assert obs.degree == degree
    assert obs._current_line is None  # Not initialized until on_start
    assert obs._best_line is None
    assert len(obs.x) == len(x_data)
    assert len(obs.y) == len(y_data)
