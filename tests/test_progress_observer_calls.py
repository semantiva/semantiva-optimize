from semantiva_optimize.progress.base import (
    ProgressObserver,
    StartEvent,
    StepEvent,
    EndEvent,
)


class Recorder(ProgressObserver):
    def __init__(self):
        self.calls = []

    def on_start(self, e):  # pragma: no cover - simple
        self.calls.append(("start", e.run_id))

    def on_step(self, e):  # pragma: no cover - simple
        self.calls.append(("step", e.run_id, e.iter, e.is_best))

    def on_best(self, e):  # pragma: no cover - simple
        self.calls.append(("best", e.run_id, e.iter))

    def on_end(self, e):  # pragma: no cover - simple
        self.calls.append(("end", e.run_id, e.reason))

    def close(self):  # pragma: no cover - simple
        self.calls.append(("close",))


def test_single_run_broadcast():
    rec = Recorder()
    rec.on_start(StartEvent(total_runs=1, run_id=None, meta={}))
    for k in range(5):
        step = StepEvent(
            iter=k,
            x=[k],
            f=float(5 - k),
            feasible=True,
            violation=0.0,
            run_id=None,
            is_best=(k % 2 == 0),
            meta={},
        )
        rec.on_step(step)
        if k % 2 == 0:
            rec.on_best(step)
    rec.on_end(EndEvent(reason="done", best_x=[1.0], best_f=0.0, run_id=None, meta={}))
    rec.close()
    assert rec.calls[0][0] == "start"
    assert rec.calls[-1][0] == "close"
