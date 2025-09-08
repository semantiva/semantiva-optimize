import pytest

try:
    from semantiva_optimize.progress.poly import PolynomialPlotObserver
    from semantiva_optimize.progress.base import StartEvent, StepEvent, EndEvent
except Exception:  # pragma: no cover - dependency missing
    pytest.skip("matplotlib missing", allow_module_level=True)


def test_poly_png_written_headless(tmp_path):
    obs = PolynomialPlotObserver(
        x_data=[0, 1, 2],
        y_data=[0, 1, 4],
        mode="file",
        out_dir=str(tmp_path),
        file_prefix="poly",
    )
    obs.on_start(StartEvent(total_runs=1, run_id=None, meta={}))
    obs.on_step(
        StepEvent(
            iter=0,
            x=[0, 1, 1],
            f=1.0,
            feasible=True,
            violation=0.0,
            run_id=None,
            is_best=True,
            meta={},
        )
    )
    obs.on_end(
        EndEvent(reason="done", best_x=[0, 1, 1], best_f=1.0, run_id=None, meta={})
    )
    assert (tmp_path / "poly_final.png").exists()
