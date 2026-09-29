"""Mathviz tool backend: JSON in -> JSON out (stdout only), PNG to traces/."""

import json
import sys
import time
from pathlib import Path


def err(msg: str, hint: str) -> dict:
    return {"ok": False, "error": msg, "retry_hint": hint}


def main() -> int:
    try:
        args = json.loads(sys.argv[1] if len(sys.argv) > 1 else "{}")
    except Exception as e:
        print(json.dumps(err(f"bad JSON: {e}",
            'Retry as {"kind":"function","title":"y=x^2","expr":"x**2"}')))
        return 0
    kind = args.get("kind", "function")
    title = str(args.get("title", "chart"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(json.dumps(err(f"import failed: {e}", "Tool error; answer in text instead.")))
        return 0
    try:
        outdir = Path("traces")
        alt = Path("../traces")
        outdir = outdir if outdir.exists() or not alt.exists() else alt
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"chart-{int(time.time() * 1000) % 10**8}.png"
        fig, ax = plt.subplots()
        if kind == "function":
            from sympy import lambdify, sympify
            from sympy.abc import x
            expr = sympify(str(args.get("expr", "x**2")), rational=True)
            f = lambdify(x, expr, "numpy")
            xmin = float(args.get("xmin", -10))
            xmax = float(args.get("xmax", 10))
            xs = np.linspace(xmin, xmax, 400)
            ax.plot(xs, f(xs))
        elif kind == "bar":
            labels = list(args.get("labels", []))
            values = [float(v) for v in args.get("values", [])]
            if not labels or len(labels) != len(values) or len(labels) > 12:
                print(json.dumps(err("labels/values must match, 1-12 items",
                    'Retry as {"kind":"bar","title":"t","labels":["A","B"],"values":[1,2]}')))
                return 0
            ax.bar(labels, values)
        elif kind == "scatter":
            xs = [float(v) for v in args.get("xs", [])]
            ys = [float(v) for v in args.get("ys", [])]
            if not xs or len(xs) != len(ys) or len(xs) > 200:
                print(json.dumps(err("xs/ys must match, 1-200 items",
                    'Retry as {"kind":"scatter","title":"t","xs":[1,2],"ys":[3,4]}')))
                return 0
            ax.scatter(xs, ys)
        else:
            print(json.dumps(err(f"unknown kind {kind}",
                'Retry as {"kind":"function","title":"y=x^2","expr":"x**2"}')))
            return 0
        ax.set_title(title)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(json.dumps({"ok": True, "png_path": str(path), "title": title}))
    except Exception as e:
        print(json.dumps(err(f"plot failed: {e}",
            'Retry as {"kind":"function","title":"y=x^2","expr":"x**2"}')))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
