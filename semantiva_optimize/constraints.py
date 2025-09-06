from typing import Callable, Sequence

# User functions: g(x) <= 0 (ineq), h(x) = 0 (eq)
Ineq = Callable[[Sequence[float]], float]
Eq   = Callable[[Sequence[float]], float]


class Constraints:
    def __init__(self, ineq: list[Ineq] | None = None, eq: list[Eq] | None = None):
        self.ineq = ineq or []
        self.eq = eq or []
