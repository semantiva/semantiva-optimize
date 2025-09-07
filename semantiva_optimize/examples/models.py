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
Example optimization models for testing and demonstration.

This module provides sample objective functions commonly used in optimization
testing, including parabolic functions and polynomial residual models.
"""

from __future__ import annotations
from typing import Sequence, Iterable


class ParabolaModel:
    """Simple parabolic objective function: f(x) = (x - x_star)^2."""

    def __init__(self, x_star: float = 3.0):
        """Initialize parabola with minimum at x_star."""
        self.x_star = float(x_star)

    def objective(self, x: Sequence[float]) -> float:
        """Compute parabolic objective function value."""
        e = float(x[0]) - self.x_star
        return e * e

    def gradient(self, x: Sequence[float]) -> list[float] | None:
        """Compute gradient of parabolic function."""
        return [2.0 * (float(x[0]) - self.x_star)]


class PolyResidualModel:
    """Polynomial residual model for root-finding via least squares."""

    def __init__(self, coeffs: Iterable[float]):
        """Initialize with polynomial coefficients (highest degree first)."""
        self.coeffs = [float(c) for c in coeffs]
        n = len(self.coeffs) - 1
        self.dcoeffs = [self.coeffs[i] * (n - i) for i in range(n)]

    @staticmethod
    def _horner(coeffs: list[float], x: float) -> float:
        """Evaluate polynomial using Horner's method."""
        v = 0.0
        for c in coeffs:
            v = v * x + c
        return v

    def objective(self, x: Sequence[float]) -> float:
        """Compute squared polynomial residual."""
        xx = float(x[0])
        p = self._horner(self.coeffs, xx)
        return p * p

    def gradient(self, x: Sequence[float]) -> list[float] | None:
        """Compute gradient of squared residual."""
        xx = float(x[0])
        p = self._horner(self.coeffs, xx)
        dp = self._horner(self.dcoeffs, xx) if self.dcoeffs else 0.0
        return [2.0 * p * dp]


def make_model(name: str, **params):
    """
    Create example model instances by name.

    Args:
        name: Model name ('parabola', 'quadratic', 'poly_residual', etc.)
        **params: Parameters passed to model constructor

    Returns:
        Model instance with objective and gradient methods

    Raises:
        ValueError: If model name is not recognized
    """
    name = (name or "").lower().strip()
    if name in {"parabola", "quadratic"}:
        return ParabolaModel(**params)
    if name in {"poly_residual", "poly", "polynomial"}:
        return PolyResidualModel(**params)
    raise ValueError(f"Unknown model name: {name}")
