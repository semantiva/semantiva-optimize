import pytest


def test_compat_floatdatatype_if_available():
    try:
        from semantiva.examples.test_utils import FloatDataType
    except Exception:
        return
    try:
        import scipy  # noqa
    except Exception:
        pytest.skip("SciPy not installed")

    from semantiva.registry import load_extensions
    from semantiva.context_processors.context_observer import _ContextObserver
    from semantiva.context_processors.context_types import ContextType
    from semantiva_optimize.processors.optimizer_processor import OptimizerContextProcessor
    from semantiva_optimize.factory import make_strategy

    load_extensions("semantiva_optimize.extension")

    class M:
        def objective(self, x): return float((x[0]-1.0)**2)
        def gradient(self, x):  return [2.0*(x[0]-1.0)]

    p = OptimizerContextProcessor()
    ctx = ContextType()
    obs = _ContextObserver()

    x0 = FloatDataType(0.0).data
    p.operate_context(context=ctx, context_observer=obs, strategy=make_strategy("local"),
                      x0=[x0], bounds=[(-10,10)], model=M(),
                      termination=None, controller=None, constraints=None, strategy_params={})
    best = ctx.get_value("optimizer.best_candidate")
    assert abs(best["x"][0]-1.0) < 1e-6
