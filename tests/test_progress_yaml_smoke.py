import yaml
import pytest
from semantiva import Pipeline, Payload
from semantiva.context_processors.context_types import ContextType
from semantiva.data_types import NoDataType
from semantiva.registry.plugin_registry import load_extensions


def test_yaml_progress_smoke(tmp_path):
    try:
        import scipy  # noqa: F401
        import matplotlib  # noqa: F401
    except ImportError:
        pytest.skip("dependencies missing")

    yaml_config = f"""
extensions: ["semantiva_optimize"]

pipeline:
  nodes:
    - processor: OptimizerContextProcessor
      parameters:
        strategy: "local"
        x0: [0.0]
        model_name: "parabola"
        model_params: {{ x_star: 3.0 }}
        termination: {{ max_evals: 20 }}
        progress:
          - class: "semantiva_optimize.progress.poly.PolynomialPlotObserver"
            kwargs:
              x_data: [0,1,2]
              y_data: [0,1,4]
              mode: "file"
              out_dir: "{tmp_path.as_posix()}"
              file_prefix: "poly"
          - class: "semantiva_optimize.progress.cost.CostCurveObserver"
            kwargs:
              mode: "file"
              out_dir: "{tmp_path.as_posix()}"
              file_prefix: "cost"
    """

    config = yaml.safe_load(yaml_config)
    load_extensions(config.get("extensions", []))
    pipeline = Pipeline(config["pipeline"]["nodes"])
    pipeline.process(Payload(data=NoDataType(), context=ContextType()))
    assert (tmp_path / "poly_final.png").exists()
    assert (tmp_path / "cost_final.png").exists()
