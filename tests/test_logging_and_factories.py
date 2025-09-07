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
from semantiva import Pipeline, Payload
from semantiva.context_processors.context_types import ContextType
from semantiva.data_types import NoDataType
from semantiva.registry.plugin_registry import load_extensions


class _StubLogger:
    """Test logger that captures messages for verification."""

    def __init__(self):
        self.messages = []

    def debug(self, msg, *args):
        self.messages.append(str(msg) % args if args else str(msg))

    def info(self, msg, *args):
        self.messages.append(str(msg) % args if args else str(msg))

    def warning(self, msg, *args):
        self.messages.append(str(msg) % args if args else str(msg))

    def error(self, msg, *args):
        self.messages.append(str(msg) % args if args else str(msg))


def test_logs_emitted_with_log_every_1():
    """Test that optimization logging works correctly with log_every=1."""
    try:
        import scipy  # noqa: F401
    except Exception:
        pytest.skip("SciPy not installed")

    # Load extension
    load_extensions(["semantiva_optimize"])

    # Create logger to capture output
    logger = _StubLogger()

    # Create pipeline node configuration with logging
    nodes = [
        {
            "processor": "OptimizerContextProcessor",
            "parameters": {
                "strategy": "local",
                "x0": [0.0],
                "bounds": [[-10, 10]],
                "model_name": "parabola",
                "model_params": {"x_star": 3.0},
                "termination": None,  # Uses default termination
                "log_every": 1,
            },
        }
    ]

    # Execute pipeline with logger
    pipeline = Pipeline(nodes, logger=logger)
    pipeline.process(Payload(data=NoDataType(), context=ContextType()))

    # Verify logging occurred
    joined = "\n".join(logger.messages)
    assert "[optimize] start" in joined
    assert "iter=" in joined
    assert "[optimize] done" in joined


def test_make_model_factory_names():
    """Test that model factory creates correct model types."""
    from semantiva_optimize.examples.models import make_model

    # Test parabola model creation
    parabola = make_model("parabola", x_star=1.5)
    assert parabola.__class__.__name__ == "ParabolaModel"

    # Test polynomial residual model creation
    poly = make_model("poly_residual", coeffs=[1.0, 0.0, -2.0])
    assert hasattr(poly, "objective") and hasattr(poly, "gradient")
