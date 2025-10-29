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

import os
from typing import Sequence, Optional

import numpy as np
import matplotlib

if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from .base import ProgressObserver, StartEvent, StepEvent, EndEvent

__all__ = ["PolynomialPlotObserver"]


class PolynomialPlotObserver(ProgressObserver):
    """Scatter plot with current-best polynomial fit."""

    def __init__(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        degree: Optional[int] = None,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        mode: str = "file",
        out_dir: str = "./_progress",
        file_prefix: str = "poly",
        dpi: int = 140,
    ) -> None:
        self.x = np.asarray(x_data, float)
        self.y = np.asarray(y_data, float)
        self.degree = degree
        self.mode = mode
        self.out_dir = out_dir
        self.file_prefix = file_prefix
        self.dpi = dpi
        os.makedirs(self.out_dir, exist_ok=True)
        if x_min is None:
            x_min = float(self.x.min())
        if x_max is None:
            x_max = float(self.x.max())
        self.grid = np.linspace(x_min, x_max, 256)
        self.fig: Optional[Figure] = None
        self.ax: Optional[Axes] = None
        self._current_line: Optional[Line2D] = (
            None  # Will hold the dashed "current" line
        )
        self._best_line: Optional[Line2D] = None  # Will hold the solid "best" line
        # Fixed, simple palette
        self._data_color = "black"
        self._fit_color = "#1f77b4"  # blue
        self._annot_color = "#ff7f0e"  # orange

    def _poly(self, theta, x):
        deg = self.degree if self.degree is not None else len(theta) - 1
        return sum(theta[k] * (x**k) for k in range(deg + 1))

    def on_start(self, e: StartEvent) -> None:  # pragma: no cover - simple
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.scatter(
            self.x, self.y, s=12, label="measurements", color=self._data_color
        )

        # Initialize both current (dashed) and best (solid) lines
        (self._current_line,) = self.ax.plot(
            [],
            [],
            linestyle="--",
            linewidth=1.2,
            label="current",
            color=self._fit_color,
        )
        (self._best_line,) = self.ax.plot(
            [], [], linewidth=2.0, label="best", color="darkblue"
        )

        self.ax.set_title("Polynomial fit — iter=0")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.legend(loc="best")
        if self.mode == "window":
            plt.ion()  # Turn on interactive mode
            self.fig.show()

    def _update_line(self, line, theta):
        """Update a line with polynomial evaluated over grid."""
        y_hat = [self._poly(theta, gx) for gx in self.grid]
        line.set_data(self.grid, y_hat)

    def on_step(self, e: StepEvent) -> None:  # pragma: no cover - tested indirectly
        # Always show the candidate being tested now (dashed)
        if self._current_line is not None:
            self._update_line(self._current_line, e.x)

        # Update title with current iteration info
        if self.ax is not None:
            self.ax.set_title(
                f"iter={e.iter}  f={e.f:.3g}  θ={[round(v, 2) for v in e.x]}"
            )

        if self.mode == "window":
            if self.fig is not None:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                # Keep window open and visible
                if (
                    hasattr(self.fig.canvas, "manager")
                    and self.fig.canvas.manager is not None
                    and hasattr(self.fig.canvas.manager, "window")
                ):
                    self.fig.canvas.manager.window.wm_deiconify()
                    self.fig.canvas.manager.window.lift()
        else:
            # Save current step as PNG
            if self.fig is not None:
                self.fig.tight_layout()
                self.fig.savefig(
                    os.path.join(self.out_dir, f"{self.file_prefix}_iter{e.iter}.png"),
                    dpi=self.dpi,
                )

    def on_best(self, e: StepEvent) -> None:  # pragma: no cover - updates best line
        # Update best-so-far (solid) when we have a new best
        if self._best_line is not None:
            self._update_line(self._best_line, e.x)

    def on_end(self, e: EndEvent) -> None:  # pragma: no cover - simple
        if self.fig is not None:
            self.fig.tight_layout()
            self.fig.savefig(
                os.path.join(self.out_dir, f"{self.file_prefix}_final.png"),
                dpi=self.dpi,
            )
            # For window mode, ensure the window stays visible
            if self.mode == "window":
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

    def close(self) -> None:  # pragma: no cover - simple
        # Don't close immediately in window mode - let the user handle it
        if self.mode != "window":
            try:
                if self.fig is not None:
                    plt.close(self.fig)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

    def force_close(self) -> None:  # pragma: no cover - simple
        """Force close the figure even in window mode."""
        try:
            if self.fig is not None:
                plt.close(self.fig)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
