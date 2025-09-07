from semantiva_optimize.strategies.local_convex import LocalConvex
from semantiva_optimize.strategies.nelder_mead import NelderMead


def make_strategy(name: str, **params):
    name = (name or "").lower()
    if name in {"local", "lbfgsb", "slsqp", "local_convex"}:
        return LocalConvex()
    if name in {"nelder", "nelder-mead", "neldermead"}:
        return NelderMead()
    raise ValueError(f"Unknown strategy: {name}")
