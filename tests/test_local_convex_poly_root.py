import math, pytest
from semantiva.registry import load_extensions
from semantiva.context_processors.context_observer import _ContextObserver
from semantiva.context_processors.context_types import ContextType

load_extensions("semantiva_optimize.extension")


# Solve p(x)=x^2-2 by minimizing L(x)=p(x)^2; provide gradient of L
class PolyModel:
    def objective(self, x):
        p = x[0]*x[0] - 2.0
        return float(p*p)
    def gradient(self, x):
        p = x[0]*x[0] - 2.0
        dLdx = 4.0*x[0]*p
        return [float(dLdx)]


def test_local_convex_finds_root_sqrt2():
    try:
        import scipy  # noqa
    except Exception:
        pytest.skip("SciPy not installed")

    from semantiva_optimize.processors.optimizer_processor import OptimizerContextProcessor
    from semantiva_optimize.factory import make_strategy
    from semantiva_optimize.termination import Termination

    p = OptimizerContextProcessor()
    ctx = ContextType()
    obs = _ContextObserver()

    p.operate_context(
        context=ctx, context_observer=obs,
        strategy=make_strategy("local"),
        x0=[0.5], bounds=[(-10, 10)],
        termination=Termination(max_evals=200, ftol_abs=1e-12, xtol_abs=1e-12),
        model=PolyModel(), controller=None, constraints=None, strategy_params={}
    )
    best = ctx.get_value("optimizer.best_candidate")
    v = best["x"][0]
    assert abs(v - math.sqrt(2.0)) < 1e-4 or abs(v + math.sqrt(2.0)) < 1e-4
    assert best["value"] < 1e-8
