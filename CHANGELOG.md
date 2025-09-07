# Changelog

All notable changes for semantiva-optimize.

## 0.1.0 — 2025-09-07

- Initial public release.
- Core components:
  - `OptimizerContextProcessor` — single entry-point for optimization runs inside Semantiva pipelines.
  - Strategies: `LocalConvex` (L-BFGS-B / SLSQP) and `NelderMead` (gradient-free).
  - `Termination` dataclass for stop criteria.
  - `Constraints`, `ModelAdapter`, `ControllerAdapter` for integrations.
  - Factory helper `make_strategy(...)` and example models for quickstarts.
- Context outputs: `optimizer.history`, `optimizer.runs`, `optimizer.best_candidate`, `optimizer.termination`.
- Included examples and pipeline wiring in `README.md` and `examples/`.

Notes
- Install: `pip install semantiva-optimize` (or use the package manager configured for your environment).
- Runtime: Some strategies require `scipy` at runtime; install if you plan to use them.
- License: Apache-2.0

No breaking changes (first release).
