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
Controller adapter interfaces for optimization.

This module defines the ControllerAdapter protocol and provides a default
NullController implementation for optimization scenarios that don't require
controller interaction.
"""

from typing import Protocol, Sequence, Any


class ControllerAdapter(Protocol):
    """Protocol for controller adapters in optimization scenarios."""

    def reset(self, seed: int | None = None) -> None:
        """Reset controller state with optional random seed."""

    def apply(self, x: Sequence[float]) -> Any:
        """Apply optimization parameters to controller."""

    def safe(self, x: Sequence[float]) -> bool:
        """Check if parameters are safe for controller."""


class NullController:
    """Default no-op controller implementation."""

    def reset(self, seed: int | None = None) -> None:
        """Reset state (no-op implementation)."""

    def apply(self, x):  # pylint: disable=unused-argument
        """Apply parameters (returns zero)."""
        return 0.0

    def safe(self, x):  # pylint: disable=unused-argument
        """Check safety (always returns True)."""
        return True
