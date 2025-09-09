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

import time
from typing import List


from semantiva_optimize.processors.optimizer_processor import OptimizerContextProcessor
from semantiva_optimize.factory import make_strategy
from semantiva_optimize.termination import Termination
from semantiva.context_processors.context_types import ContextType
from semantiva.context_processors.context_observer import _ContextObserver


class Parabola:
    def objective(self, x):
        return float((x[0] - 3.0) ** 2)

    def gradient(self, x):
        return [2.0 * (x[0] - 3.0)]


class Recorder:
    def __init__(self):
        self.calls: List[tuple] = []

    def on_start(self, e):
        self.calls.append(("start", e.run_id, e.total_runs))

    def on_step(self, e):
        self.calls.append(("step", e.run_id, e.iter, e.is_best))

    def on_best(self, e):
        self.calls.append(("best", e.run_id, e.iter))

    def on_end(self, e):
        self.calls.append(("end", e.run_id, e.reason))

    def close(self):
        self.calls.append(("close",))


def _run(proc, **kwargs):
    ctx, obs = ContextType(), _ContextObserver()
    proc.operate_context(context=ctx, context_observer=obs, **kwargs)
    return obs.observer_context


def test_events_contract_and_order():
    rec = Recorder()
    proc = OptimizerContextProcessor()
    ctx = _run(
        proc,
        strategy=make_strategy("local"),
        x0=[0.0],
        bounds=[(-10, 10)],
        termination=Termination(max_evals=10),
        model=Parabola(),
        controller=None,
        constraints=None,
        progress=[rec],
    )
    assert rec.calls[0][0] == "start"
    assert rec.calls[-1][0] == "close"
    # Ensure best events appear in sequence
    seq = [c[0] for c in rec.calls]
    assert seq.index("best") > seq.index("step")
    assert ctx.get_value("optimizer.history")


def test_processor_broadcasts_and_noop_when_none():
    rec = Recorder()
    proc = OptimizerContextProcessor()
    ctx1 = _run(
        proc,
        strategy=make_strategy("local"),
        x0=[0.5],
        bounds=[(-10, 10)],
        termination=Termination(max_evals=10),
        model=Parabola(),
        controller=None,
        constraints=None,
        progress=[rec],
    )
    best1 = ctx1.get_value("optimizer.best_candidate")
    ctx2 = _run(
        proc,
        strategy=make_strategy("local"),
        x0=[0.5],
        bounds=[(-10, 10)],
        termination=Termination(max_evals=10),
        model=Parabola(),
        controller=None,
        constraints=None,
        progress=None,
    )
    best2 = ctx2.get_value("optimizer.best_candidate")
    assert abs(best1["value"] - best2["value"]) < 1e-8
    assert rec.calls[0][0] == "start"


def test_scipy_callbacks_stream_intermediate_steps():
    rec = Recorder()
    proc = OptimizerContextProcessor()
    _run(
        proc,
        strategy=make_strategy("local"),
        x0=[5.0],
        bounds=[(-10, 10)],
        termination=Termination(max_evals=30),
        model=Parabola(),
        controller=None,
        constraints=None,
        progress=[rec],
    )
    step_calls = [c for c in rec.calls if c[0] == "step"]
    assert len(step_calls) > 1


def test_progress_update_every_skips_steps():
    rec = Recorder()
    proc = OptimizerContextProcessor()
    _run(
        proc,
        strategy=make_strategy("local"),
        x0=[5.0],
        bounds=[(-10, 10)],
        termination=Termination(max_evals=30),
        model=Parabola(),
        controller=None,
        constraints=None,
        progress=[rec],
        progress_update_every=2,
    )
    steps = [c for c in rec.calls if c[0] == "step"]
    assert 1 <= len(steps) < 20


def test_progress_throttle_limits_rate(monkeypatch):
    times = [0.0, 0.05, 0.10, 0.12, 0.70, 0.76]
    it = iter(times)
    monkeypatch.setattr(time, "time", lambda: next(it))
    rec = Recorder()
    proc = OptimizerContextProcessor()
    _run(
        proc,
        strategy=make_strategy("local"),
        x0=[5.0],
        bounds=[(-10, 10)],
        termination=Termination(max_evals=30),
        model=Parabola(),
        controller=None,
        constraints=None,
        progress=[rec],
        progress_throttle_s=0.5,
    )
    steps = [c for c in rec.calls if c[0] == "step"]
    assert len(steps) <= 3


def test_observer_exceptions_are_swallowed():
    class Bad:
        def on_start(self, e):
            raise RuntimeError

        def on_step(self, e):
            raise RuntimeError

        def on_best(self, e):
            raise RuntimeError

        def on_end(self, e):
            raise RuntimeError

        def close(self):
            raise RuntimeError

    bad = Bad()
    proc = OptimizerContextProcessor()
    ctx = _run(
        proc,
        strategy=make_strategy("local"),
        x0=[1.0],
        bounds=[(-10, 10)],
        termination=Termination(max_evals=10),
        model=Parabola(),
        controller=None,
        constraints=None,
        progress=[bad],
    )
    best = ctx.get_value("optimizer.best_candidate")
    assert best


def test_multi_start_run_ids_and_total_runs():
    rec = Recorder()
    proc = OptimizerContextProcessor()
    _run(
        proc,
        strategy=make_strategy("local"),
        x0=[5.0],
        bounds=[(-10, 10)],
        termination=Termination(max_evals=30),
        model=Parabola(),
        controller=None,
        constraints=None,
        progress=[rec],
        multi_start=[[5.0], [0.0]],
    )
    starts = [c for c in rec.calls if c[0] == "start"]
    assert starts[0][2] == 2
    steps = [c for c in rec.calls if c[0] == "step"]
    run_ids = {c[1] for c in steps}
    assert run_ids == {0, 1}


def test_no_side_effect_on_solver_path():
    proc = OptimizerContextProcessor()
    ctx1 = _run(
        proc,
        strategy=make_strategy("local"),
        x0=[6.0],
        bounds=[(-10, 10)],
        termination=Termination(max_evals=30),
        model=Parabola(),
        controller=None,
        constraints=None,
        progress=None,
    )
    best1 = ctx1.get_value("optimizer.best_candidate")
    rec = Recorder()
    ctx2 = _run(
        proc,
        strategy=make_strategy("local"),
        x0=[6.0],
        bounds=[(-10, 10)],
        termination=Termination(max_evals=30),
        model=Parabola(),
        controller=None,
        constraints=None,
        progress=[rec],
    )
    best2 = ctx2.get_value("optimizer.best_candidate")
    assert abs(best1["value"] - best2["value"]) < 1e-8


def test_strategy_progress_broadcaster_backwards_compatibility():
    class Simple:
        def tell(self, x0, model, bounds, constraints, termination, **kwargs):
            return make_strategy("local").tell(
                x0=x0,
                model=model,
                bounds=bounds,
                constraints=constraints,
                termination=termination,
            )

    proc = OptimizerContextProcessor()
    _run(
        proc,
        strategy=Simple(),
        x0=[0.0],
        bounds=[(-10, 10)],
        termination=Termination(max_evals=10),
        model=Parabola(),
        controller=None,
        constraints=None,
        progress=None,
    )


def test_history_and_observer_in_lockstep():
    rec = Recorder()
    proc = OptimizerContextProcessor()
    ctx = _run(
        proc,
        strategy=make_strategy("local"),
        x0=[0.0],
        bounds=[(-10, 10)],
        termination=Termination(max_evals=30),
        model=Parabola(),
        controller=None,
        constraints=None,
        progress=[rec],
    )
    best = ctx.get_value("optimizer.best_candidate")
    hist = ctx.get_value("optimizer.history")
    assert hist
    assert abs(hist[-1]["value"] - best["value"]) < 1e-10
    assert rec.calls[0][0] == "start" and rec.calls[-1][0] == "close"
