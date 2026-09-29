"""SymPy tool backend: JSON in -> JSON out (stdout only)."""

import json
import sys


def err(msg: str, hint: str) -> dict:
    return {"ok": False, "error": msg, "retry_hint": hint}


def main() -> int:
    try:
        args = json.loads(sys.argv[1] if len(sys.argv) > 1 else "{}")
    except Exception as e:
        print(json.dumps(err(f"bad JSON: {e}", 'Retry as {"op":"eval","expression":"23*47"}')))
        return 0
    op = args.get("op", "solve")
    try:
        from sympy import S, diff, factorint, integrate, isprime, latex, solveset, sympify
        from sympy.abc import x
        from sympy.physics.units import convert_to, foot, kilogram, kilometer, liter, meter, mile, pound, quart
    except Exception as e:
        print(json.dumps(err(f"import failed: {e}", "Tool error; do not retry math, explain instead.")))
        return 0
    try:
        if op == "eval":
            expr = sympify(str(args.get("expression", "")), rational=True)
            val = expr.doit()
            print(json.dumps({"ok": True, "result": str(val), "latex": latex(val)}))
        elif op == "solve":
            eq = sympify(str(args.get("equation", "")), rational=True)
            sols = solveset(eq, x, domain=S.Reals)
            print(json.dumps({"ok": True, "result": str(sols), "latex": latex(sols)}))
        elif op == "diff":
            f = sympify(str(args.get("expression", "")), rational=True)
            d = diff(f, x)
            print(json.dumps({"ok": True, "result": str(d), "latex": latex(d)}))
        elif op == "integrate":
            f = sympify(str(args.get("expression", "")), rational=True)
            res = integrate(f, x)
            print(json.dumps({"ok": True, "result": str(res), "latex": latex(res)}))
        elif op == "ntheory":
            n = int(args.get("n", 0))
            if abs(n) > 10**9:
                print(json.dumps(err("n too large", 'Retry with |n|<=1e9 as {"op":"ntheory","n":42}')))
            else:
                print(json.dumps({
                    "ok": True,
                    "result": f"prime={bool(isprime(n))} factors={factorint(abs(n)) if n else {}}",
                }))
        elif op == "units":
            from_amounts = {
                "km_to_mi": lambda v: v * 1000 * meter,
                "mi_to_km": lambda v: v * mile,
                "m_to_ft": lambda v: v * meter,
                "kg_to_lb": lambda v: v * kilogram,
                "l_to_quart": lambda v: v * liter,
            }
            targets = {
                "km_to_mi": mile, "mi_to_km": kilometer, "m_to_ft": foot,
                "kg_to_lb": pound, "l_to_quart": quart,
            }
            key = str(args.get("conversion", ""))
            val = float(args.get("value", 1))
            if key not in from_amounts:
                if key in ("c_to_f", "f_to_c", "c_to_k"):
                    if key == "c_to_f":
                        out = val * 9 / 5 + 32
                        print(json.dumps({"ok": True, "result": f"{val}C = {out}F"}))
                    elif key == "f_to_c":
                        out = (val - 32) * 5 / 9
                        print(json.dumps({"ok": True, "result": f"{val}F = {out}C"}))
                    else:
                        out = val + 273.15
                        print(json.dumps({"ok": True, "result": f"{val}C = {out}K"}))
                else:
                    print(json.dumps(err(f"unknown conversion {key}",
                        'Retry as {"op":"units","conversion":"c_to_f","value":100} (keys: '
                        + ",".join(sorted(list(from_amounts) + ["c_to_f", "f_to_c", "c_to_k"])) + ")")))
            else:
                out = convert_to(from_amounts[key](val), targets[key])
                print(json.dumps({"ok": True, "result": str(out)}))
        else:
            print(json.dumps(err(f"unknown op {op}",
                'Retry as {"op":"eval","expression":"23*47"}')))
    except Exception as e:
        print(json.dumps(err(f"compute failed: {e}",
            'Retry as {"op":"eval","expression":"23*47"} or rephrase the equation')))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
