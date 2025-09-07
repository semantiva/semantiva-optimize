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
Constraint definitions for optimization problems.

This module defines constraint types and the Constraints class that holds
inequality and equality constraint functions for optimization strategies.
"""

from typing import Callable, Sequence

# User functions: g(x) <= 0 (ineq), h(x) = 0 (eq)
Ineq = Callable[[Sequence[float]], float]
Eq = Callable[[Sequence[float]], float]


class Constraints:
    """
    Container for optimization constraints.

    Holds lists of inequality (g(x) <= 0) and equality (h(x) = 0) constraint functions.
    """

    def __init__(self, ineq: list[Ineq] | None = None, eq: list[Eq] | None = None):
        """
        Initialize constraint container.

        Args:
            ineq: List of inequality constraint functions g(x) <= 0
            eq: List of equality constraint functions h(x) = 0
        """
        self.ineq = ineq or []
        self.eq = eq or []
