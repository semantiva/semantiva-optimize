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

from semantiva_optimize.config_preprocessor import optimize_config_preprocessor


def test_config_preprocessor_transforms():
    node = {
        "processor": "OptimizerContextProcessor",
        "parameters": {
            "strategy": "local",
            "strategy_params": {"alpha": 1},
            "termination": {"max_evals": 10},
            "controller": {"type": "imaging.sim", "params": {"seed": 1}},
            "constraints": {
                "bounds": [[-1, 1]],
                "ineq": [{"type": "linear", "a": [1], "b": 0}],
            },
        },
    }
    out = optimize_config_preprocessor(node)
    params = out["parameters"]
    assert params["strategy"]["class"].endswith("LocalConvex")
    assert params["bounds"] == [[-1, 1]]
    assert params["constraints"]["class"].endswith("LinearConstraints")
