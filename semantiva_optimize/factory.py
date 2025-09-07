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
Factory functions for creating optimization strategy instances.

This module provides the main entry point for creating optimization strategies
by name, with aliases for common patterns.
"""

from semantiva_optimize.strategies.local_convex import LocalConvex
from semantiva_optimize.strategies.nelder_mead import NelderMead


def make_strategy(name: str, **params):  # pylint: disable=unused-argument
    """
    Create an optimization strategy instance by name.

    Args:
        name: Strategy name (supports aliases like 'local', 'nelder-mead')
        **params: Currently unused, reserved for future strategy configuration

    Returns:
        Strategy instance (LocalConvex or NelderMead)

    Raises:
        ValueError: If strategy name is not recognized
    """
    name = (name or "").lower()
    if name in {"local", "lbfgsb", "slsqp", "local_convex"}:
        return LocalConvex()
    if name in {"nelder", "nelder-mead", "neldermead"}:
        return NelderMead()
    raise ValueError(f"Unknown strategy: {name}")
