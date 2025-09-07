from semantiva.registry import SemantivaExtension
from semantiva.registry.class_registry import ClassRegistry


class OptimizeExtension(SemantivaExtension):
    def register(self) -> None:
        ClassRegistry.register_modules([
            "semantiva_optimize.processors.optimizer_processor",
            "semantiva_optimize.strategies.local_convex",
            "semantiva_optimize.strategies.nelder_mead",
            "semantiva_optimize.adapters.model_adapter",
            "semantiva_optimize.adapters.controller_adapter",
            "semantiva_optimize.termination",
            "semantiva_optimize.constraints",
            "semantiva_optimize.factory",
        ])
