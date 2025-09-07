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

from semantiva_optimize.param_resolvers import optimize_param_resolver


def test_param_resolver_strategy():
    out = optimize_param_resolver("opt.strategy:nelder-mead")
    assert out["class"].endswith("NelderMead")


def test_param_resolver_termination():
    out = optimize_param_resolver("opt.termination:max_evals=123,ftol_abs=1e-9")
    assert out["class"].endswith("Termination")
    assert out["kwargs"]["max_evals"] == 123
    assert out["kwargs"]["ftol_abs"] == 1e-9


def test_param_resolver_controller():
    out = optimize_param_resolver("opt.controller:pkg.MyController(seed=7)")
    assert out["class"] == "pkg.MyController"
    assert out["kwargs"]["seed"] == 7
