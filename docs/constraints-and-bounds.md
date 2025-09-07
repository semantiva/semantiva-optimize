# Constraints and Bounds

Inequalities follow the convention `g(x) = A·x - b ≤ 0` and equalities
`h(x) = A·x - b = 0`.

Bounds are specified per dimension as `[[lo, hi], ...]` and are forwarded to
strategies that support them.

Example:

```yaml
constraints:
  bounds:
    - [-10, 10]
  ineq:
    - {type: linear, a: [1, 0], b: 0}   # x0 ≤ 0
  eq:
    - {type: linear, a: [1, -1], b: 0}  # x0 = x1
```
