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

"""Live demo for polynomial progress observers with current & best curves."""

import os
import time
os.environ.setdefault("MPLBACKEND", "TkAgg")

#from semantiva.registry import load_extensions

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

    
class FallbackLeastSquaresObjective:
    def __init__(self, family, x_data, y_data):
        self.family = family
        self.x_data = x_data
        self.y_data = y_data
        self.calls = 0
        
    def objective(self, theta):
        """Minimize residual for fitting y = x^2 with theta=[c0, c1, c2, c3]."""
        self.calls += 1
        time.sleep(0.05)  # Small delay to see progress
        
        # Compute residual sum of squares
        residual_sum = 0.0
        for x_val, y_val in zip(self.x_data, self.y_data):
            y_pred = sum(theta[k] * (x_val ** k) for k in range(len(theta)))
            residual_sum += (y_val - y_pred) ** 2
            
        print(f"Function call {self.calls}: theta={[round(t, 3) for t in theta]}, f={residual_sum:.6f}")
        return residual_sum

class FallbackPolynomialFamily:
    def __init__(self, degree):
        self.degree = degree
        

LeastSquaresObjective = FallbackLeastSquaresObjective  # type: ignore[misc]
PolynomialFamily = FallbackPolynomialFamily  # type: ignore[misc]


class SlowProgressObserver:
    """Progress observer that shows updates clearly."""

    def __init__(self, name):
        self.name = name

    def on_start(self, e):
        print(f"\n=== {self.name}: Optimization Started ===")
        print(f"Total runs: {e.total_runs}")

    def on_step(self, e):
        print(f"{self.name}: Step {e.iter} - x={[round(xi, 3) for xi in e.x]}, f={e.f:.6f}, is_best={e.is_best}")

    def on_best(self, e):
        print(f"{self.name}: NEW BEST! Step {e.iter} - f={e.f:.6f}")

    def on_end(self, e):
        print(f"{self.name}: Optimization ended - {e.reason}, best_f={e.best_f:.6f}")

    def close(self):
        print(f"{self.name}: Closed")


# Main demo
print("=== Live Polynomial Progress Observer Demo ===")
print("This demo fits a cubic polynomial to noisy data.")
print("You'll see live matplotlib windows showing current (dashed) and best (solid) fits.")
print()

# Synthetic data approximately quadratic with noise
x_data = [0, 1, 2, 3, 4, 5]
y_data = [0.1, 1.2, 4.1, 8.9, 15.7, 24.9]

degree = 3  # Explicit degree for plotting AND validation

# Validate degree/theta alignment
x0 = [0.0, 0.0, 1.0, 0.0]  # c0, c1, c2, c3 (cubic polynomial coefficients)
assert len(x0) == degree + 1, (
    f"x0 length ({len(x0)}) must equal degree+1 ({degree+1}) for polynomial plotting"
)

# Create the polynomial fitting objective
objective = LeastSquaresObjective(
    family=PolynomialFamily(degree=degree),
    x_data=x_data, 
    y_data=y_data
)

# Create observers with explicit degree and window mode
poly_obs = PolynomialPlotObserver(
    x_data=x_data, 
    y_data=y_data,
    degree=degree,        # Ensure curved polynomial is plotted
    mode="window",        # Use 'file' if headless
    file_prefix="live_poly"
)
cost_obs = CostCurveObserver(mode="window", file_prefix="live_cost")
debug_obs = SlowProgressObserver("DEBUG")

p = OptimizerContextProcessor()
ctx, obs = ContextType(), _ContextObserver()

print("Starting polynomial fitting optimization...")
print("You should see matplotlib windows showing:")
print("- Black dots: measurements")
print("- Dashed line: current candidate polynomial") 
print("- Solid line: best-so-far polynomial")
print()

p.operate_context(
    context=ctx,
    context_observer=obs,
    strategy=make_strategy("local"),  # L-BFGS-B works well for least squares
    x0=x0,
    bounds=None,
    termination=Termination(max_evals=150, ftol_abs=1e-10),
    model=objective,
    controller=None,
    constraints=None,
    progress=[poly_obs, cost_obs, debug_obs],
    progress_update_every=1,
    progress_throttle_s=0.0,
    strategy_params={},
)

print("\n=== Optimization Complete ===")
print("Best:", ctx.get_value("optimizer.best_candidate"))

# Keep windows open for viewing
print("\nWindows should now be visible. Close them or press Enter to exit...")
try:
    input()
except KeyboardInterrupt:
    pass

print("Closing windows...")
poly_obs.force_close()
cost_obs.force_close()
debug_obs.close()

print("Demo complete!")
