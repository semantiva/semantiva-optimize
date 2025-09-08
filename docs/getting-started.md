# YAML-first Optimizer

`semantiva-optimize` lets you configure optimization pipelines entirely in YAML.
The config preprocessor turns friendly blocks into runtime descriptors – no
Python glue required.

```yaml
strategy: "local"           # resolved to LocalConvex
termination: {max_evals: 200}
controller:
  type: "imaging.sim"
  params: {seed: 42}
constraints:
  ineq:
    - {type: linear, a: [-1], b: 0}
```

All conversions happen through the regular Semantiva resolver stack; YAML stays
pure data.

Want live visuals? See [Progress Observers](progress-observers.md) for cost curves and real-time polynomial overlays.
