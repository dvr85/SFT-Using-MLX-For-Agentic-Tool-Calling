# Chart chooser

| Kind | Args | Use when |
|---|---|---|
| `function` | `expr` (in x, e.g. `x**2`), `xmin/xmax` | graph a curve |
| `bar` | `labels[]`, `values[]` (equal length, ≤12) | compare categories |
| `scatter` | `xs[]`, `ys[]` (equal length, ≤200) | show relationship |

All kinds accept `title`. Output PNG is `1200x800` equivalent (dpi 150).
Retry hints are machine-readable: fix the named field and call once more.
