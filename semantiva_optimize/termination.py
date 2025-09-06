from dataclasses import dataclass


@dataclass
class Termination:
    max_evals: int = 200
    ftol_abs: float = 1e-9
    ftol_rel: float = 1e-9
    xtol_abs: float = 1e-9
