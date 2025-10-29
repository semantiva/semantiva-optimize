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

"""Test polynomial observer titles and line rendering."""

import os


def test_polynomial_observer_title_and_lines(tmp_path, monkeypatch):
    """Test that observer updates titles with iter/f and maintains current & best lines."""
    # Set headless backend
    monkeypatch.setenv("MPLBACKEND", "Agg")

    from semantiva_optimize.progress.poly import PolynomialPlotObserver
    from semantiva_optimize.progress.base import StartEvent, StepEvent, EndEvent

    x_data = [0, 1, 2, 3, 4]
    y_data = [0, 1, 4, 9, 16]
    degree = 2

    obs = PolynomialPlotObserver(
        x_data=x_data,
        y_data=y_data,
        degree=degree,
        mode="file",
        out_dir=str(tmp_path),
        file_prefix="test",
    )

    # Start event
    start_event = StartEvent(total_runs=1, run_id=None, meta={"strategy": "test"})
    obs.on_start(start_event)

    # Check that figure and lines are initialized
    assert obs.fig is not None
    assert obs.ax is not None
    assert obs._current_line is not None
    assert obs._best_line is not None

    # Check initial title
    assert "iter=0" in obs.ax.get_title()

    # Simulate some optimization steps
    theta1 = [1.0, 2.0, 0.5]  # First candidate
    step_event1 = StepEvent(
        iter=1,
        x=theta1,
        f=10.5,
        feasible=True,
        violation=0.0,
        run_id=None,
        is_best=True,
        meta={"strategy": "test"},
    )

    obs.on_step(step_event1)

    # Check title updated with iteration info
    title = obs.ax.get_title()
    assert "iter=1" in title
    assert "f=10.5" in title
    assert str(theta1) in title or "θ=" in title

    # Best event should update the best line
    obs.on_best(step_event1)

    # Test another step (not best)
    theta2 = [0.8, 1.5, 0.7]  # Different candidate
    step_event2 = StepEvent(
        iter=2,
        x=theta2,
        f=12.3,
        feasible=True,
        violation=0.0,
        run_id=None,
        is_best=False,
        meta={"strategy": "test"},
    )

    obs.on_step(step_event2)

    # Title should reflect new iteration
    title = obs.ax.get_title()
    assert "iter=2" in title
    assert "f=12.3" in title

    # End event
    end_event = EndEvent(
        reason="completed",
        best_x=theta1,
        best_f=10.5,
        run_id=None,
        meta={"strategy": "test"},
    )
    obs.on_end(end_event)

    # Check that final PNG was saved
    final_files = list(tmp_path.glob("test_final.png"))
    assert len(final_files) == 1

    obs.close()


def test_polynomial_lines_separate_current_and_best():
    """Test that current and best lines are maintained separately."""
    # Set headless backend
    os.environ["MPLBACKEND"] = "Agg"

    from semantiva_optimize.progress.poly import PolynomialPlotObserver
    from semantiva_optimize.progress.base import StartEvent, StepEvent

    x_data = [0, 1, 2]
    y_data = [0, 1, 4]

    obs = PolynomialPlotObserver(x_data=x_data, y_data=y_data, degree=2, mode="file")

    # Initialize
    start_event = StartEvent(total_runs=1, run_id=None, meta={})
    obs.on_start(start_event)

    # Get initial line data (should be empty)
    current_data_before = obs._current_line.get_data()
    best_data_before = obs._best_line.get_data()

    # Should start with empty lines
    assert len(current_data_before[0]) == 0
    assert len(best_data_before[0]) == 0

    # First step (becomes best)
    theta1 = [0.0, 0.0, 1.0]
    step1 = StepEvent(
        iter=1,
        x=theta1,
        f=5.0,
        feasible=True,
        violation=0.0,
        run_id=None,
        is_best=True,
        meta={},
    )

    obs.on_step(step1)  # Updates current line
    obs.on_best(step1)  # Updates best line

    # Both lines should now have data
    current_data_after = obs._current_line.get_data()
    best_data_after = obs._best_line.get_data()

    assert len(current_data_after[0]) > 0
    assert len(best_data_after[0]) > 0

    # Second step (not best, only updates current)
    theta2 = [1.0, 1.0, 0.5]
    step2 = StepEvent(
        iter=2,
        x=theta2,
        f=8.0,
        feasible=True,
        violation=0.0,
        run_id=None,
        is_best=False,
        meta={},
    )

    obs.on_step(step2)  # Updates current line only

    # Current line should have new data, best line unchanged
    current_data_new = obs._current_line.get_data()
    best_data_unchanged = obs._best_line.get_data()

    # Current line changed
    assert not all(x == y for x, y in zip(current_data_after[1], current_data_new[1]))
    # Best line unchanged
    assert all(x == y for x, y in zip(best_data_after[1], best_data_unchanged[1]))

    obs.close()


def test_polynomial_poly_function():
    """Test the internal _poly function evaluates polynomials correctly."""
    os.environ["MPLBACKEND"] = "Agg"

    from semantiva_optimize.progress.poly import PolynomialPlotObserver

    obs = PolynomialPlotObserver(x_data=[0, 1, 2], y_data=[0, 1, 4], degree=2)

    # Test quadratic: f(x) = 1 + 2x + 3x^2
    theta = [1.0, 2.0, 3.0]

    # f(0) = 1 + 2*0 + 3*0^2 = 1
    assert obs._poly(theta, 0) == 1.0

    # f(1) = 1 + 2*1 + 3*1^2 = 6
    assert obs._poly(theta, 1) == 6.0

    # f(2) = 1 + 2*2 + 3*4 = 17
    assert obs._poly(theta, 2) == 17.0
