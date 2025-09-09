# Progress Observers

Attach real-time, read-only observers to `OptimizerContextProcessor`:

```yaml
progress:
  - class: "semantiva_optimize.progress.poly.PolynomialPlotObserver"
    kwargs:
      x_data: "{{ context.traces.t }}"
      y_data: "{{ context.traces.sigma_x }}"
      mode: "file"          # 'file' saves PNGs; 'window' shows a live plot
      out_dir: "./_progress"
      file_prefix: "poly"
  - class: "semantiva_optimize.progress.cost.CostCurveObserver"
    kwargs:
      mode: "file"
      out_dir: "./_progress"
      file_prefix: "cost"
progress_update_every: 2
progress_throttle_s: 0.05
```

* **PolynomialPlotObserver**: data scatter + current-best polynomial curve, with a “now testing” annotation.
* **CostCurveObserver**: best-so-far objective value vs iteration; works for single and multi-start.
* Observers are **read-only** (they never affect the optimization).
* Headless by default (`Agg` backend); for live windows set a GUI backend (e.g., `TkAgg`).
* The same payload sent to observers is appended to `optimizer.history` during the run, keeping history and observers in lockstep. Inequality constraints follow the convention `g(x) ≤ 0`.

See also:

* `examples/polynomial_progress.yaml` – headless PNGs
* `semantiva_optimize/examples/polynomial_progress_live.py` – live matplotlib window
