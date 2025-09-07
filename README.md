# semantiva-optimize

Optimization extension for [Semantiva]. It provides a generic **OptimizerContextProcessor** and pluggable strategies.
Currently available strategies:

* **Local Convex** – L-BFGS-B for bounds; SLSQP with simple constraints
* **Nelder-Mead** – gradient-free simplex search

## Install

```bash
pip install "semantiva-optimize[scipy]"
```

## Quick Start (Python)

```python
from semantiva.registry import load_extensions
from semantiva.context_processors.context_observer import _ContextObserver
from semantiva.context_processors.context_types import ContextType

load_extensions("semantiva_optimize.extension")

from semantiva_optimize.processors.optimizer_processor import OptimizerContextProcessor
from semantiva_optimize.factory import make_strategy

# Simple quadratic: min (x-3)^2
class QuadraticModel:
    def objective(self, x): return float((x[0]-3.0)**2)
    def gradient(self, x):  return [2.0*(x[0]-3.0)]

p = OptimizerContextProcessor()
ctx = ContextType()
obs = _ContextObserver()

p.operate_context(
    context=ctx, context_observer=obs,
    strategy=make_strategy("local"),
    x0=[0.0], bounds=[(-100, 100)],
    termination=None, model=QuadraticModel(),
    controller=None, constraints=None, strategy_params={}
)

print(ctx.get_value("optimizer.best_candidate"))
# → {'x': [2.999999...], 'value': ~0.0, 'feasible': True, 'meta': {}}
```

### Nelder-Mead (gradient-free)

```python
class PolyModel:
    def objective(self, x):
        p = x[0]*x[0] - 2.0
        return float(p*p)
    def gradient(self, x):
        return None

p = OptimizerContextProcessor()
ctx, obs = ContextType(), _ContextObserver()
p.operate_context(
    context=ctx, context_observer=obs,
    strategy=make_strategy("nelder-mead"),
    x0=[0.1], bounds=None,
    termination=Termination(max_evals=500, ftol_abs=1e-12, xtol_abs=1e-12),
    model=PolyModel(), controller=None, constraints=None, strategy_params={},
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
    x0=[0.0], multi_start=starts,
    bounds=[(-10, 10)], termination=Termination(max_evals=100),
    model=QuadraticModel(), controller=None, constraints=None, strategy_params={},
)
print(ctx.get_value("optimizer.runs"))      # three independent runs
print(ctx.get_value("optimizer.best_candidate"))
```

### Pipeline YAML wiring

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
        strategy: "local"
        x0: [0.0]             # seed (unused when multi_start provided)
        multi_start: "{{ slice.values.starts }}"
        bounds: [[-10, 10]]
        termination: { max_evals: 100, ftol_abs: 1e-10, xtol_abs: 1e-10 }
        strategy_params: {}
```

## Context Outputs

* `optimizer.strategy: str`
* `optimizer.params: dict` *(bounds, termination, strategy params)*
* `optimizer.history: list[dict]` *(MEU-like step records; kept in context)*
* `optimizer.runs: list[{x,value,feasible,meta}]` *(results for each start)*
* `optimizer.best_candidate: {"x": [...], "value": float, "feasible": bool, "meta": {...}}`
* `optimizer.termination: {"reason": str, "metrics": dict, "budget": dict}`

## Notes

* Iteration details are recorded inside the **context**. Semantiva's core trace remains node-level.
* When SciPy is not installed, Local Convex will no-op; tests are skipped.

## License

Apache-2.0
