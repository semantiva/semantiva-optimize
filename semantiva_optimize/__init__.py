# Copyright 2025 Semantiva authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Semantiva Optimize - A first-class optimizer fully integrated into Semantiva.

This package provides an OptimizerContextProcessor that can tune simulation parameters,
fit models, calibrate instruments, or run controller-in-the-loop experiments with
epistemic transparency by design.
"""


from semantiva.registry import SemantivaExtension
from semantiva.registry.class_registry import ClassRegistry

# Import main components for convenient access
from . import strategies
from . import processors


class SemantivaOptimize(SemantivaExtension):
    """Extension class for registering Semantiva Optimize modules."""

    def register(self) -> None:
        """Register all optimization-related modules with the Semantiva class registry."""
        ClassRegistry.register_modules(
            [
                "semantiva_optimize.processors",
                "semantiva_optimize.strategies",
                "semantiva_optimize.adapters",
                "semantiva_optimize.termination",
                "semantiva_optimize.constraints",
                "semantiva_optimize.factory",
                "semantiva_optimize.examples.models",
            ]
        )


__all__ = [
    "processors",
    "strategies",
    "SemantivaOptimize",
]
