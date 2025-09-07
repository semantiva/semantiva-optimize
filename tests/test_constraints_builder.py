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
from semantiva_optimize.descriptors import LinearConstraints


def test_linear_constraints_builder():
    cons = LinearConstraints(
        bounds=[[-1, 1], [0, 2]],
        ineq=[{"type": "linear", "a": [1, 0], "b": 0}],
        eq=[{"type": "linear", "a": [1, -1], "b": 0}],
    )
    assert cons.bounds == [[-1, 1], [0, 2]]
    assert cons.ineq[0]([0.5, 0.2])[0] == pytest.approx(0.5)
