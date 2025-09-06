# semantiva-optimize

Optimization extension for [Semantiva]. It provides a generic **OptimizerContextProcessor** and pluggable strategies.  
**First release includes one strategy:** Local Convex (L-BFGS-B for bounds; SLSQP with simple constraints).

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

## Context Outputs

* `optimizer.strategy: str`
* `optimizer.params: dict` *(bounds, termination, strategy params)*
* `optimizer.history: list[dict]` *(MEU-like step records; kept in context)*
* `optimizer.best_candidate: {"x": [...], "value": float, "feasible": bool, "meta": {...}}`
* `optimizer.termination: {"reason": str, "metrics": dict, "budget": dict}`

## Notes

* Iteration details are recorded inside the **context**. Semantiva's core trace remains node-level.
* When SciPy is not installed, Local Convex will no-op; tests are skipped.

## License

Apache-2.0
