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
Termination criteria for optimization algorithms.

This module defines the Termination dataclass that specifies when
optimization should stop based on various convergence criteria.
"""

from dataclasses import dataclass


@dataclass
class Termination:
    """
    Optimization termination criteria.

    Defines when optimization should stop based on maximum evaluations
    and various tolerance thresholds for convergence detection.
    """

    max_evals: int = 200
    ftol_abs: float = 1e-9
    ftol_rel: float = 1e-9
    xtol_abs: float = 1e-9
