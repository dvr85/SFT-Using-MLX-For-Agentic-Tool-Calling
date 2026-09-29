use crate::store::SessionStore;
use serde_json::Value;
use std::time::Duration;

/// Closed tool registry v3: solve_symbolic | plot_chart (SKILL-backed).
/// Each tool shells to its skill script; stdout JSON only enters context.
fn run_script(program: &str, script: &str, args: &Value, timeout_s: u64) -> String {
    let json = serde_json::to_string(args).unwrap_or_else(|_| "{}".to_string());
    let mut cmd = std::process::Command::new(program);
    if program == "uv" {
        cmd.args(["run", "python", script, &json]);
    } else {
        cmd.args([script, &json]);
    }
    // Run from sft_harness/ so relative skills/ and traces/ paths resolve.
    if let Ok(dir) = std::env::current_dir() {
        if dir.file_name().is_some_and(|n| n != "sft_harness") {
            let p = dir.join("sft_harness");
            if p.exists() {
                cmd.current_dir(p);
            }
        }
    }
    // Timeout: best-effort via wait_timeout pattern (no extra deps).
    use std::io::Read;
    let mut child = match cmd.stdout(std::process::Stdio::piped()).spawn() {
        Ok(c) => c,
        Err(e) => return format!("Tool error: spawn failed: {e}."),
    };
    let deadline = std::time::Instant::now() + Duration::from_secs(timeout_s.max(1));
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut out = String::new();
                if let Some(mut so) = child.stdout.take() {
                    let _ = so.read_to_string(&mut out);
                }
                if !status.success() {
                    return format!("Tool error: script exit {status}. Output: {}.", out.trim());
                }
                return format_script_result(&out);
            }
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    return "Tool error: timed out. Retry with a smaller input.".to_string();
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => return format!("Tool error: {e}."),
        }
    }
}

fn format_script_result(out: &str) -> String {
    let v: Value = match serde_json::from_str(out.trim()) {
        Ok(v) => v,
        Err(_) => return format!("Tool error: bad script output: {}.", out.trim()),
    };
    if v.get("ok").and_then(|b| b.as_bool()).unwrap_or(false) {
        let mut parts = vec![];
        for k in ["result", "latex", "png_path", "title"] {
            if let Some(s) = v.get(k).and_then(|x| x.as_str()) {
                if !s.is_empty() {
                    parts.push(format!("{k}: {s}"));
                }
            }
        }
        if parts.is_empty() {
            v.to_string()
        } else {
            parts.join("\n")
        }
    } else {
        let e = v.get("error").and_then(|x| x.as_str()).unwrap_or("failed");
        let h = v.get("retry_hint").and_then(|x| x.as_str()).unwrap_or("");
        if h.is_empty() {
            format!("Invalid arguments: {e}.")
        } else {
            format!("Invalid arguments: {e} {h}")
        }
    }
}

/// Dispatch validated args; all errors become strings (never raised).
pub fn dispatch(name: &str, args: &Value, store: &SessionStore, timeout_s: u64) -> String {
    match name {
        "solve_symbolic" => {
            if !args.is_object() {
                return "Invalid arguments: expected JSON object. Retry as {\"op\":\"eval\",\"expression\":\"23*47\"}.".to_string();
            }
            run_script("uv", "skills/sympy/scripts/solve.py", args, timeout_s)
        }
        "plot_chart" => {
            if !args.is_object() {
                return "Invalid arguments: expected JSON object. Retry as {\"kind\":\"function\",\"title\":\"y=x^2\",\"expr\":\"x**2\"}.".to_string();
            }
            run_script("uv", "skills/mathviz/scripts/plot.py", args, timeout_s)
        }
        "todo_write" => {
            let op = args.get("op").and_then(|v| v.as_str()).unwrap_or("");
            if op.is_empty() {
                return "Invalid arguments: op must be create|update|complete.".to_string();
            }
            let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
            store.todo(op, content)
        }
        _ => format!("Unknown tool: {name}. Available: solve_symbolic, plot_chart, todo_write."),
    }
}
