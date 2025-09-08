import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import matplotlib

if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .base import ProgressObserver, StartEvent, StepEvent, EndEvent

__all__ = ["CostCurveObserver"]


class CostCurveObserver(ProgressObserver):
    """Plot best-so-far cost vs iteration."""

    def __init__(
        self,
        mode: str = "file",
        out_dir: str = "./_progress",
        file_prefix: str = "cost",
        dpi: int = 140,
    ) -> None:
        self.mode = mode
        self.out_dir = out_dir
        self.file_prefix = file_prefix
        self.dpi = dpi
        os.makedirs(self.out_dir, exist_ok=True)
        self.fig: Optional[Figure] = None
        self.ax: Optional[Axes] = None
        self.best_by_run: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self.last_best: Dict[int, float] = {}

        # Simple, fixed color palette for consistent live plots
        self._main_color = "#1f77b4"  # blue for main progress
        self._run_color = "#7f7f7f"  # gray for additional runs
        self._best_color = "#ff7f0e"  # orange for current best marker

    def on_start(self, e: StartEvent) -> None:  # pragma: no cover - simple
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 4))
            self.ax.set_title("Cost vs iteration")
            self.ax.set_xlabel("iteration")
            self.ax.set_ylabel("best f so far")
            if self.mode == "window":
                plt.ion()  # Turn on interactive mode
                self.fig.show()
        # Initialize with starting point assumption
        for run_id in range(e.total_runs):
            rid = run_id if e.total_runs > 1 else -1
            self.last_best[rid] = float("inf")

    def on_step(self, e: StepEvent) -> None:  # pragma: no cover - tested indirectly
        rid = e.run_id if e.run_id is not None else -1

        # Always track cost progression, not just when is_best
        if rid not in self.last_best:
            self.last_best[rid] = float("inf")

        # Update best value if this is better
        if e.f < self.last_best[rid]:
            self.last_best[rid] = e.f
            is_new_best = True
        else:
            is_new_best = False

        # Add point to plot data - show all iterations, highlight best
        self.best_by_run[rid].append((e.iter, self.last_best[rid]))

        if self.ax:
            # Clear existing lines
            for line in self.ax.lines[:]:
                line.remove()

            for k, pts in sorted(self.best_by_run.items()):
                if not pts:  # Skip empty runs
                    continue
                xs, ys = zip(*pts)
                label = f"run {k}" if k != -1 else "optimization progress"
                color = self._main_color if k == -1 else self._run_color
                # Use consistent marker/line styling instead of the default color cycle
                self.ax.plot(
                    xs,
                    ys,
                    marker="o",
                    linestyle="-",
                    color=color,
                    label=label,
                    alpha=0.9,
                    linewidth=2,
                    markersize=6,
                )

            # Highlight current point if it's a new best
            # Highlight current point if it's a new best
            if is_new_best:
                self.ax.scatter(
                    [e.iter],
                    [e.f],
                    color=self._best_color,
                    edgecolor="white",
                    s=70,
                    zorder=5,
                    label="current best" if e.iter > 0 else None,
                )

            # self.ax.legend()
            self.ax.grid(True, alpha=0.3)

            if self.mode == "file":
                if self.fig is not None:
                    self.fig.tight_layout()
                    self.fig.savefig(
                        os.path.join(
                            self.out_dir, f"{self.file_prefix}_iter{e.iter}.png"
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
