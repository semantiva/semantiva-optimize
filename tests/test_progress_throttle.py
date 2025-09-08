import pytest

try:
    from semantiva_optimize.progress.cost import CostCurveObserver
    from semantiva_optimize.progress.base import StartEvent, StepEvent, EndEvent
except Exception:  # pragma: no cover - dependency missing
    pytest.skip("matplotlib missing", allow_module_level=True)


def test_throttle_does_not_crash(tmp_path):
    obs = CostCurveObserver(mode="file", out_dir=str(tmp_path), file_prefix="cost")
    obs.on_start(StartEvent(total_runs=1, run_id=None, meta={}))
    for k in range(20):
        obs.on_step(
            StepEvent(
                iter=k,
                x=[k],
                f=float(20 - k),
                feasible=True,
                violation=0.0,
                run_id=None,
                is_best=True,
                meta={},
            )
        )
    obs.on_end(EndEvent(reason="done", best_x=[0.0], best_f=0.0, run_id=None, meta={}))
    assert (tmp_path / "cost_final.png").exists()
