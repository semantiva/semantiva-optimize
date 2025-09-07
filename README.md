# Semantiva Optimize

Optimization extension for Semantiva. It provides a generic **OptimizerContextProcessor** and pluggable strategies.

Currently available strategies:

- **Local Convex** — L-BFGS-B (bounds), SLSQP (simple constraints)
- **Nelder–Mead** — gradient-free simplex search

> Works anywhere you can provide an `objective(x)` (and optionally `gradient(x)`), with optional constraints, bounds, and multi-start fan-in.

---

## Install

```bash
pip install semantiva-optimize
```

---

## Quick Start (Python)

```python
from semantiva.registry import load_extensions
from semantiva.context_processors.context_observer import _ContextObserver
from semantiva.context_processors.context_types import ContextType

# If your environment doesn't auto-load entry points, do it explicitly:
load_extensions("semantiva_optimize.extension")

from semantiva_optimize.processors.optimizer_processor import OptimizerContextProcessor
from semantiva_optimize.factory import make_strategy
from semantiva_optimize.termination import Termination

# Simple quadratic: min (x-3)^2
class QuadraticModel:
    def objective(self, x): return float((x[0] - 3.0) ** 2)
    def gradient(self, x):  return [2.0 * (x[0] - 3.0)]

p = OptimizerContextProcessor()
ctx, obs = ContextType(), _ContextObserver()

p.operate_context(
    context=ctx, context_observer=obs,
    strategy=make_strategy("local"),     # aliases: "local", "local-convex"
    x0=[0.0],
    bounds=[(-100.0, 100.0)],
    termination=Termination(max_evals=200, ftol_abs=1e-12, xtol_abs=1e-12),
    model=QuadraticModel(),
    controller=None,
    constraints=None,
    strategy_params={},
)

print(ctx.get_value("optimizer.best_candidate"))
# → {'x': [2.999999...], 'value': ~0.0, 'feasible': True, 'meta': {}}
```

### Nelder–Mead (gradient-free)

```python
from semantiva_optimize.termination import Termination

class PolyModel:
    def objective(self, x):
        p = x[0] * x[0] - 2.0
        return float(p * p)
    def gradient(self, x):
        return None  # intentionally None

p = OptimizerContextProcessor()
ctx, obs = ContextType(), _ContextObserver()

p.operate_context(
    context=ctx, context_observer=obs,
    strategy=make_strategy("nelder-mead"),  # alias: "nelder_mead"
    x0=[0.1],
    bounds=None,
    termination=Termination(max_evals=500, ftol_abs=1e-12, xtol_abs=1e-12),
    model=PolyModel(),
    controller=None,
    constraints=None,
    strategy_params={},
)

print(ctx.get_value("optimizer.best_candidate"))
```

### Constraints (quick example)

```python
from semantiva_optimize.constraints import Constraints

# Inequality: x[0] >= 0  →  ineq(x) = x[0] - 0 >= 0
def ineq(x): return [x[0] - 0.0]

cons = Constraints(ineq=[ineq], eq=[])

p = OptimizerContextProcessor()
ctx, obs = ContextType(), _ContextObserver()

p.operate_context(
    context=ctx, context_observer=obs,
    strategy=make_strategy("local"),
    x0=[-1.0],
    bounds=[(-10.0, 10.0)],
    termination=Termination(max_evals=100),
    model=QuadraticModel(),
    controller=None,
    constraints=cons,
    strategy_params={},
)
print(ctx.get_value("optimizer.best_candidate"))
```

### Multi-start execution

```python
starts = [[-5.0], [0.0], [10.0]]

p = OptimizerContextProcessor()
ctx, obs = ContextType(), _ContextObserver()

p.operate_context(
    context=ctx, context_observer=obs,
    strategy=make_strategy("local"),
    x0=[0.0],                   # seed (unused when multi_start provided)
    multi_start=starts,         # list of x0 vectors
    bounds=[(-10.0, 10.0)],
    termination=Termination(max_evals=100),
    model=QuadraticModel(),
    controller=None,
    constraints=None,
    strategy_params={},
)

print(ctx.get_value("optimizer.runs"))        # three independent runs
print(ctx.get_value("optimizer.best_candidate"))
```

---

## Pipeline YAML wiring

```yaml
pipeline:
  nodes:
    - processor: DataSlicerProcessor
      parameters:
        values:
          starts:
            - [-5.0]
            - [0.0]
            - [10.0]

    - processor: OptimizerContextProcessor
      parameters:
        strategy: "local"                 # aliases: local, local-convex
        x0: [0.0]                         # seed (ignored if multi_start provided)
        multi_start: "{{ slice.values.starts }}"
        bounds: [[-10.0, 10.0]]
        termination: { max_evals: 100, ftol_abs: 1e-10, xtol_abs: 1e-10 }
        strategy_params: {}
```

Run with `-v` to enable DEBUG logs:

```bash
semantiva run examples/quadratic_local.yaml -v
```

Example log lines:

```
[optimize] start strategy=LocalConvex bounds=[(-10, 10)] x0=[0]
[optimize] iter=0 x=[0] f=9 feas=True viol=0
[optimize] done reason=completed best_x=[3] best_f=0
```

**Field meanings**

* `iter` — internal iteration counter
* `x` — current candidate
* `f` — objective value
* `feas` — whether safety/constraints are satisfied
* `viol` — max constraint violation observed

See `examples/` for:

* `quadratic_local.yaml`: minimize $f(x) = (x-3)^2$
* `polyroot_local.yaml`: find a root of $x^2 - 2 = 0$ by minimizing the squared residual

---

## API surface (at a glance)

**`OptimizerContextProcessor.operate_context(...)` key parameters**

* `strategy`: result of `make_strategy(name, **strategy_params)`
  * Accepted names: `"local"`, `"local-convex"`, `"nelder-mead"`, `"nelder_mead"`
* `x0`: list/array initial guess
* `multi_start`: list of `x0` (each a list/array); if provided, runs independently and selects best
* `bounds`: list of `(low, high)` per dimension, or `None`
* `termination`: `Termination(max_evals, ftol_abs, xtol_abs, ...)`
* `model`: object with `objective(x) -> float` and optional `gradient(x) -> array | None`
* `controller`: optional adapter with `reset/apply/safe` (for HIL/sim)
* `constraints`: `Constraints(ineq=[fn,...], eq=[fn,...])` with each `fn(x) -> list/array`
* `strategy_params`: dict passed to the chosen strategy

---

## Context outputs

* `optimizer.strategy: str`
* `optimizer.params: dict` *(bounds, termination, strategy params)*
* `optimizer.history: list[dict]` *(step records kept in context)*
* `optimizer.runs: list[{x, value, feasible, meta}]` *(one per start)*
* `optimizer.best_candidate: {"x": [...], "value": float, "feasible": bool, "meta": {...}}`
* `optimizer.termination: {"reason": str, "metrics": dict, "budget": dict}`

> Iteration details are recorded inside the **context**. Semantiva’s core trace remains node-level.

---

## License

Apache-2.0