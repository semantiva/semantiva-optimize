"""Live demo for progress observers."""

import os
import time

os.environ.setdefault("MPLBACKEND", "TkAgg")


# load_extensions("semantiva_optimize")

# The following imports must occur after calling `load_extensions` because
# the registry may install entry points used by these modules. Mark them
# as noqa: E402 (module level import not at top of file) to satisfy ruff.
from semantiva.context_processors.context_observer import _ContextObserver  # noqa: E402
from semantiva.context_processors.context_types import ContextType  # noqa: E402

from semantiva_optimize.processors.optimizer_processor import (
    OptimizerContextProcessor,
)  # noqa: E402
from semantiva_optimize.factory import make_strategy  # noqa: E402
from semantiva_optimize.progress.poly import PolynomialPlotObserver  # noqa: E402
from semantiva_optimize.progress.cost import CostCurveObserver  # noqa: E402
from semantiva_optimize.termination import Termination  # noqa: E402


class SlowQuadraticModel:
    """Quadratic model with artificial delay to show progress."""

    def __init__(self):
        self.calls = 0

    def objective(self, x):
        self.calls += 1
        # Add a small delay to make progress visible
        time.sleep(0.1)
        result = float((x[0] - 3.0) ** 2)
        print(f"Function call {self.calls}: x={x[0]:.3f}, f={result:.6f}")
        return result

    def gradient(self, x):
        return [2.0 * (x[0] - 3.0)]


class SlowProgressObserver:
    """Progress observer that shows updates clearly."""

    def __init__(self, name):
        self.name = name

    def on_start(self, e):
        print(f"\n=== {self.name}: Optimization Started ===")
        print(f"Total runs: {e.total_runs}")

    def on_step(self, e):
        print(f"{self.name}: Step {e.iter} - x={e.x}, f={e.f:.6f}, is_best={e.is_best}")

    def on_best(self, e):
        print(f"{self.name}: NEW BEST! Step {e.iter} - f={e.f:.6f}")

    def on_end(self, e):
        print(f"{self.name}: Optimization ended - {e.reason}, best_f={e.best_f:.6f}")

    def close(self):
        print(f"{self.name}: Closed")


print("=== Live Progress Observer Demo ===")
print("This demo will show live matplotlib windows during optimization.")
print("The optimization will be deliberately slow to make progress visible.")
print()

x_data = [0, 1, 2, 3, 4, 5]
y_data = [0.1, 1.2, 4.1, 8.9, 15.7, 24.9]

# Create observers with window mode
poly_obs = PolynomialPlotObserver(
    x_data=x_data, y_data=y_data, mode="window", file_prefix="poly_live"
)
cost_obs = CostCurveObserver(mode="window", file_prefix="cost_live")
debug_obs = SlowProgressObserver("DEBUG")

p = OptimizerContextProcessor()
ctx, obs = ContextType(), _ContextObserver()

print("Starting optimization...")
print("You should see matplotlib windows appear during optimization.")
print()

p.operate_context(
    context=ctx,
    context_observer=obs,
    strategy=make_strategy("nelder-mead"),
    x0=[0.0],
    bounds=None,
    termination=Termination(
        max_evals=20, ftol_abs=1e-10
    ),  # Fewer evals for quicker demo
    model=SlowQuadraticModel(),
    controller=None,
    constraints=None,
    progress=[poly_obs, cost_obs, debug_obs],
    progress_update_every=1,
    progress_throttle_s=0.0,
    strategy_params={},
)

print("\n=== Optimization Complete ===")
print("Best:", obs.observer_context.get_value("optimizer.best_candidate"))

# Keep windows open for a bit longer
print("\nWindows should now be visible and will stay open.")
print("Keeping windows open for 5 seconds...")
time.sleep(5)

print("Closing windows...")
poly_obs.force_close()
cost_obs.force_close()
debug_obs.close()

print("Demo complete!")
