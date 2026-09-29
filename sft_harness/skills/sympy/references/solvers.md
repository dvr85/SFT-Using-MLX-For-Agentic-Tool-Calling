# Solver guide

| Solver | Use when | Returns |
|---|---|---|
| `solveset(eq, x, domain)` | algebraic equations (preferred) | set |
| `linsolve` / `nonlinsolve` | linear / nonlinear systems | set of tuples |
| `dsolve` | ODEs (hidden subcommand for now) | Eq |
| `nsolve` | no closed form, need numeric | float |
| `diff` / `integrate` / `limit` | calculus | exact expr |
| `factorint` / `isprime` | number theory | factors / bool |
| `eval` | plain arithmetic `23*47`, `0.1+0.2` | exact via Rational |

Pattern: solve → verify by substitution (`simplify(expr.subs(x, sol)) == 0`).
Edge cases: empty set (no solution), `whole=0` in ratios (error, not inf),
non-integral floats for integer paths (reject with retry hint).
