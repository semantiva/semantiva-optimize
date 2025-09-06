import pytest
from semantiva.registry import load_extensions
from semantiva.context_processors.context_observer import _ContextObserver
from semantiva.context_processors.context_types import ContextType

load_extensions("semantiva_optimize.extension")


class QuadraticModel:
    def objective(self, x): return float((x[0]-3.0)**2)
    def gradient(self, x):  return [2.0*(x[0]-3.0)]


@pytest.mark.parametrize("x0", [[0.0],[10.0],[2.0]])
def test_local_convex_converges_to_3(x0):
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
        x0=x0, bounds=[(-100, 100)],
        termination=Termination(max_evals=100, ftol_abs=1e-12, xtol_abs=1e-12),
        model=QuadraticModel(), controller=None, constraints=None, strategy_params={}
    )
    best = ctx.get_value("optimizer.best_candidate")
    assert abs(best["x"][0] - 3.0) < 1e-6
    assert best["value"] < 1e-12
