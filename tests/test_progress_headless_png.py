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
