from semantiva_optimize.strategies.local_convex import LocalConvex


def make_strategy(name: str, **params):
    name = (name or "").lower()
    if name in {"local", "lbfgsb", "slsqp", "local_convex"}:
        s = LocalConvex()
    else:
        raise ValueError(f"Unknown strategy: {name}")
    return s
