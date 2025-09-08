# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Real-time progress observers and integration with `OptimizerContextProcessor`:
  - `ProgressObserver` protocol and event dataclasses (`StartEvent`, `StepEvent`, `EndEvent`).
  - `PolynomialPlotObserver` — polynomial fit visualization with "now testing" annotation.
  - `CostCurveObserver` — best-so-far cost vs. iteration curve and optional file/window modes.
  - Throttling and update-frequency controls for observers (`progress_throttle_s`, `progress_update_every`).
  - Config preprocessor / param-resolver support for progress descriptors and YAML aliases.

### Changed
- Strategies that use SciPy (e.g. `NelderMead`, local convex strategies) now expose intermediate iterations via callbacks so progress observers receive multiple step events during optimization.

### Fixed
- Various matplotlib compatibility and typing issues so observers work in both headless (Agg) and interactive backends.

## [0.1.0] - 2025-09-07

### Added
- Initial public release.
- Core components:
  - `OptimizerContextProcessor` — single entry point for optimization runs inside Semantiva pipelines.
  - Strategies: `LocalConvex` (L-BFGS-B / SLSQP) and `NelderMead` (gradient-free).
  - `Termination` dataclass for stop criteria.
  - `Constraints`, `ModelAdapter`, `ControllerAdapter` for integrations.
  - Factory helper `make_strategy(...)` and example models for quickstarts.
- Context outputs: `optimizer.history`, `optimizer.runs`, `optimizer.best_candidate`, `optimizer.termination`.
- Included examples and pipeline wiring in `README.md` and `examples/`.

### Notes
- Install: `pip install semantiva-optimize` (or use the package manager configured for your environment).
- Runtime: Some strategies require `scipy` at runtime; install if you plan to use them.
- License: Apache-2.0
