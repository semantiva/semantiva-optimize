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
