from typing import Protocol, Sequence, Any


class Controller(Protocol):
    def reset(self, seed: int | None = None) -> None: ...
    def apply(self, x: Sequence[float]) -> Any: ...
    def safe(self, x: Sequence[float]) -> bool: ...


class NullController:
    def reset(self, seed: int | None = None) -> None:
        ...

    def apply(self, x):
        return 0.0

    def safe(self, x):
        return True
