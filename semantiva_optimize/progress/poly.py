import os
from typing import Sequence, Optional

import numpy as np
import matplotlib

if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

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
        # Fixed, simple palette
        self._data_color = "black"
        self._fit_color = "#1f77b4"  # blue
        self._annot_color = "#ff7f0e"  # orange

    def _poly(self, theta, x):
        deg = self.degree if self.degree is not None else len(theta) - 1
        return sum(theta[k] * (x**k) for k in range(deg + 1))

    def on_start(self, e: StartEvent) -> None:  # pragma: no cover - simple
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.scatter(self.x, self.y, s=12, label="data", color=self._data_color)
        self.ax.set_title("Polynomial fit — current best")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.legend(loc="best")
        if self.mode == "window":
            plt.ion()  # Turn on interactive mode
            self.fig.show()

    def _annotate_now_testing(self, e: StepEvent):  # pragma: no cover - simple
        if self.ax is None:
            return
        txt = f"iter={e.iter}  f={e.f:.4g}"
        if e.run_id is not None:
            txt += f"  run={e.run_id}"
        if len(e.x) <= 4:
            txt += "  θ=[" + ", ".join(f"{v:.3g}" for v in e.x) + "]"
        for a in list(self.ax.texts):
            if getattr(a, "_tag", None) == "now":
                a.remove()
        ann = self.ax.text(
            0.02, 0.98, txt, transform=self.ax.transAxes, va="top", ha="left"
        )
        ann._tag = "now"  # type: ignore[attr-defined]

    def on_step(self, e: StepEvent) -> None:  # pragma: no cover - tested indirectly
        if not e.is_best:
            return
        y_hat = [self._poly(e.x, gx) for gx in self.grid]
        if self.ax:
            # Remove previous "current" lines
            # Iterate a copy of lines to avoid modifying the list while iterating
            for existing_line in list(self.ax.lines):
                if existing_line.get_label() == "current":
                    existing_line.remove()

            # Plot current best with fixed color to avoid color cycling
            self.ax.plot(
                self.grid,
                y_hat,
                label="current",
                color=self._fit_color,
                linewidth=2,
            )
            self._annotate_now_testing(e)
            if self.mode == "file":
                if self.fig is not None:
                    self.fig.tight_layout()
                    self.fig.savefig(
                        os.path.join(
                            self.out_dir, f"{self.file_prefix}_best_iter{e.iter}.png"
                        ),
                        dpi=self.dpi,
                    )
            else:  # pragma: no cover - requires GUI backend
                if self.fig is not None:
                    self.fig.tight_layout()
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

    def on_best(self, e: StepEvent) -> None:  # pragma: no cover - noop
        pass

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
