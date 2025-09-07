# Controllers

Controllers can be referenced in YAML using a short alias:

```yaml
controller:
  type: "imaging.sim"
  params: {seed: 7}
```

The alias is resolved to `semantiva_imaging.controllers.SimController` by the
config preprocessor. You can also provide an explicit Python path:

```yaml
controller: {class: "my.pkg.Controller", kwargs: {}}
```
