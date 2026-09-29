use crate::config::Config;
use crate::executor;
use crate::parser::{self, ToolCall};
use crate::store::SessionStore;
use crate::trace::Tracer;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Msg {
    pub role: String,
    pub content: String,
}

/// System (tool defs) + first user turn — same semantics as src/eval.py:_prompt_messages.
pub fn prompt_messages(messages: &[Msg]) -> Vec<Msg> {
    let mut prompt: Vec<Msg> = messages
        .iter()
        .filter(|m| m.role == "system")
        .take(1)
        .cloned()
        .collect();
    if let Some(u) = messages.iter().find(|m| m.role == "user") {
        prompt.push(u.clone());
    }
    prompt
}

pub struct Agent {
    pub cfg: Config,
    client: reqwest::blocking::Client,
    store: SessionStore,
}

fn temp_store() -> SessionStore {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    SessionStore::open(
        &std::env::temp_dir().join(format!("sft-agent-{}-{n}.redb", std::process::id())),
    )
    .expect("temp store")
}

impl Agent {
    pub fn new(cfg: Config) -> Self {
        Self::with_store(cfg, temp_store())
    }

    pub fn with_store(cfg: Config, store: SessionStore) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(cfg.request_timeout_s))
            .build()
            .unwrap();
        Self { cfg, client, store }
    }

    /// Body for POST {model_url}/chat/completions. The `model` value must be the
    /// id the server was started with -- the server resolves it as a repo id,
    /// so a placeholder (e.g. "sft-harness") 404s on huggingface.co.
    /// The `tools` array switches the MiniCPM chat template into tool mode;
    /// without it the model muses about lacking live data instead of calling.
    fn request_body(&self, messages: &[Msg]) -> serde_json::Value {
        serde_json::json!({
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": self.cfg.max_tokens_per_turn,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "chat_template_kwargs": {"enable_thinking": self.cfg.enable_thinking},
            "tools": [
                {"type": "function", "function": {
                    "name": "solve_symbolic",
                    "description": "Exact symbolic math via SymPy: solve equations, calculus, matrices, number theory, unit conversion, arithmetic eval. Returns result plus LaTeX.",
                    "parameters": {"type": "object", "required": ["op"],
                        "properties": {
                            "op": {"type": "string", "enum": ["solve", "eval", "diff", "integrate", "ntheory", "units"]},
                            "equation": {"type": "string", "description": "e.g. x**2-5*x+6"},
                            "expression": {"type": "string", "description": "e.g. 23*47 or sin(x**2)"},
                            "n": {"type": "integer"},
                            "conversion": {"type": "string", "description": "e.g. c_to_f"},
                            "value": {"type": "number"}}}}},
                {"type": "function", "function": {
                    "name": "plot_chart",
                    "description": "Mathematical visualizing via matplotlib: graph a function or draw a bar/scatter chart. Returns a PNG path; embed it verbatim.",
                    "parameters": {"type": "object", "required": ["kind", "title"],
                        "properties": {
                            "kind": {"type": "string", "enum": ["function", "bar", "scatter"]},
                            "title": {"type": "string"},
                            "expr": {"type": "string", "description": "e.g. x**2"},
                            "xmin": {"type": "number"},
                            "xmax": {"type": "number"},
                            "labels": {"type": "array"},
                            "values": {"type": "array"},
                            "xs": {"type": "array"},
                            "ys": {"type": "array"}}}}},
                {"type": "function", "function": {
                    "name": "todo_write",
                    "description": "Maintain the task plan across turns. op=create replaces the plan, op=update appends an item, op=complete marks all items done. Returns the current plan.",
                    "parameters": {"type": "object", "required": ["op", "content"],
                        "properties": {
                            "op": {"type": "string", "enum": ["create", "update", "complete"]},
                            "content": {"type": "string", "description": "Plan item text."}}}}},
            ],
        })
    }

    /// mlx_lm.server (0.31.x) puts Think-mode output in `message.reasoning`
    /// (separate from `message.content`) and native tool calls in
    /// `message.tool_calls[]` ({function:{name, arguments}}). Combine all three
    /// so generation is never silently dropped; native calls are re-expressed
    /// as ToolACE bracket text for the shared parser.
    fn extract_text(msg: &serde_json::Value) -> String {
        let mut parts: Vec<String> = Vec::new();
        if let Some(calls) = msg.get("tool_calls").and_then(|v| v.as_array()) {
            for call in calls {
                let func = call.get("function").unwrap_or(call);
                let name = func.get("name").and_then(|v| v.as_str()).unwrap_or("");
                if name.is_empty() {
                    continue;
                }
                let args = func.get("arguments").unwrap_or(&serde_json::Value::Null);
                // Arguments arrive either as a JSON string (OpenAI shape) or an
                // object; normalize to k=v pairs for the ToolACE bracket parser
                // (JSON scalars render directly: "Paris" parses as a string).
                let obj: Option<serde_json::Map<String, serde_json::Value>> = match args {
                    serde_json::Value::String(s) => serde_json::from_str(s).unwrap_or(None),
                    serde_json::Value::Object(m) => Some(m.clone()),
                    _ => None,
                };
                let args_str = obj
                    .map(|m| {
                        m.iter()
                            .map(|(k, v)| format!("{k}={v}"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    })
                    .unwrap_or_default();
                parts.push(format!("[{name}({args_str})]"));
            }
        }
        for key in ["reasoning", "content"] {
            if let Some(t) = msg.get(key).and_then(|v| v.as_str()) {
                let t = t.trim();
                if !t.is_empty() {
                    parts.push(t.to_string());
                }
            }
        }
        parts.join("\n")
    }

    /// Strip <think> blocks from the final answer (trace keeps raw).
    pub fn clean_answer(s: &str) -> String {
        let mut out = s.to_string();
        while let Some(a) = out.find("<think>") {
            if let Some(b) = out[a..].find("</think>") {
                out.replace_range(a..a + b + 8, "");
            } else {
                out.truncate(a);
                break;
            }
        }
        out.trim().to_string()
    }

    fn chat_once(&self, messages: &[Msg]) -> Result<String, String> {
        let body = self.request_body(messages);
        let resp: serde_json::Value = self
            .client
            .post(format!(
                "{}/chat/completions",
                self.cfg.model_url.trim_end_matches('/')
            ))
            .json(&body)
            .send()
            .map_err(|e| format!("Tool error: {e}."))?
            .json()
            .map_err(|e| format!("Tool error: {e}."))?;
        Ok(Self::extract_text(&resp["choices"][0]["message"]))
    }

    /// ReAct loop. Returns (final_answer, steps).
    fn drive<F>(
        &self,
        mut messages: Vec<Msg>,
        mut chat: F,
        mut tracer: Option<&mut Tracer>,
    ) -> (String, usize)
    where
        F: FnMut(&[Msg]) -> Result<String, String>,
    {
        let mut prev: Option<ToolCall> = None;
        for _ in 0..self.cfg.max_steps {
            let out = match chat(&messages) {
                Ok(o) => o,
                Err(e) => return (e, messages.len()),
            };
            if out.trim().is_empty() {
                if let Some(t) = tracer.as_mut() {
                    t.log("assistant", &out);
                }
                return ("Stopped: empty response.".to_string(), messages.len());
            }
            if let Some(t) = tracer.as_mut() {
                t.log("assistant", &out);
            }
            let calls = match parser::parse_tool_calls(&out) {
                Ok(c) => c,
                Err(_) => return (Self::clean_answer(&out), messages.len()),
            };
            if prev.as_ref().is_some_and(|p| *p == calls[0]) {
                return ("Stopped: repeated tool call.".to_string(), messages.len());
            }
            prev = Some(calls[0].clone());
            messages.push(Msg {
                role: "assistant".into(),
                content: out,
            });
            for call in &calls {
                let result = executor::execute(
                    call,
                    &self.cfg.allowlist,
                    &self.store,
                    self.cfg.tool_timeout_s,
                );
                if let Some(t) = tracer.as_mut() {
                    t.log("tool", &result);
                }
                messages.push(Msg {
                    role: "tool".into(),
                    content: result,
                });
            }
        }
        (String::from("Stopped: max steps exceeded."), messages.len())
    }

    pub fn run(&self, prompt: &str, tracer: &mut Tracer) -> (String, usize) {
        let system = "You are a beginner-friendly tutor. Use tools subtly and only when needed: for multi-step tasks, first call todo_write to track the plan; call solve_symbolic for equations, calculus, units, or exact arithmetic; call plot_chart when the user asks for a graph, plot, or chart and embed its PNG path verbatim; otherwise answer directly from conversation history. After a tool result, think through it and explain simply in English: what it is, a tiny example, why it matters. Cite only the tool results you received; if results lack the answer say so plainly and never invent papers or links. Never emit ASCII-art diagrams.";
        // Strip invocation keyword if present (e.g. leading "sft-harness-mlx").
        let prompt = prompt
            .strip_prefix("sft-harness-mlx")
            .map(str::trim)
            .unwrap_or(prompt);
        let messages = vec![
            Msg {
                role: "system".into(),
                content: system.into(),
            },
            Msg {
                role: "user".into(),
                content: prompt.into(),
            },
        ];
        tracer.log("system", system);
        tracer.log("user", prompt);
        self.drive(messages, |m| self.chat_once(m), Some(tracer))
    }

    pub fn run_with<F>(&self, prompt: &str, mut fake: F) -> (String, usize)
    where
        F: FnMut(&[Msg]) -> String,
    {
        let messages = prompt_messages(&[
            Msg {
                role: "system".into(),
                content: "sys".into(),
            },
            Msg {
                role: "user".into(),
                content: prompt.into(),
            },
        ]);
        self.drive(messages, |m| Ok(fake(m)), None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_shape() {
        let msgs = vec![
            Msg {
                role: "system".into(),
                content: "s".into(),
            },
            Msg {
                role: "user".into(),
                content: "u1".into(),
            },
            Msg {
                role: "user".into(),
                content: "u2".into(),
            },
        ];
        let p = prompt_messages(&msgs);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].role, "system");
        assert_eq!(p[1].content, "u1");
    }

    #[test]
    fn request_body_uses_configured_model_id() {
        // Regression: a hardcoded placeholder ("sft-harness") made mlx_lm.server
        // resolve the body model on huggingface.co -> 404 Repository Not Found.
        let a = Agent::new(Config::default());
        let body = a.request_body(&[Msg {
            role: "user".into(),
            content: "hi".into(),
        }]);
        assert_eq!(body["model"], serde_json::json!(Config::default().model));
        assert_ne!(body["model"], serde_json::json!("sft-harness"));
    }

    #[test]
    fn request_body_sends_tools_array() {
        // Without `tools`, the MiniCPM template stays in chat mode and muses
        // about lacking live data instead of emitting tool calls.
        let a = Agent::new(Config::default());
        let body = a.request_body(&[]);
        let tools = body["tools"].as_array().expect("tools array");
        let names: Vec<&str> = tools
            .iter()
            .filter_map(|t| t["function"]["name"].as_str())
            .collect();
        for want in ["solve_symbolic", "plot_chart", "todo_write"] {
            assert!(names.contains(&want), "missing tool {want}");
        }
        // Regression guard: every required key must exist in properties (visualize bug).
        for tool in tools {
            let req = tool["function"]["required"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            for r in req {
                let key = r.as_str().unwrap_or("");
                assert!(
                    tool["function"]["properties"].get(key).is_some(),
                    "tool {} required key {key} missing from properties",
                    tool["function"]["name"]
                );
            }
        }
        assert_eq!(
            body["chat_template_kwargs"]["enable_thinking"],
            serde_json::json!(true)
        );
    }

    #[test]
    fn extract_text_prefers_reasoning_when_content_absent() {
        // Captured mlx_lm.server 0.31.x shape: Think-mode text in `reasoning`,
        // no `content` key at all, finish_reason=length.
        let msg = serde_json::json!({
            "role": "assistant",
            "reasoning": "Okay, the user is asking for the weather in Paris."
        });
        let text = Agent::extract_text(&msg);
        assert!(text.contains("weather in Paris"));
    }

    #[test]
    fn extract_text_combines_reasoning_and_content() {
        let msg = serde_json::json!({
            "role": "assistant",
            "reasoning": "User wants weather.",
            "content": "[get_weather(city=\"Paris\")]"
        });
        let text = Agent::extract_text(&msg);
        assert!(text.contains("[get_weather(city=\"Paris\")]"));
    }

    #[test]
    fn extract_text_maps_native_tool_calls_to_brackets() {
        let msg = serde_json::json!({
            "role": "assistant",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"city\": \"Paris\"}"
                }
            }]
        });
        let text = Agent::extract_text(&msg);
        let calls = parser::parse_tool_calls(&text).expect("parseable");
        assert_eq!(calls[0].name, "get_weather");
    }

    #[test]
    fn drive_reports_empty_response_as_failure() {
        let a = Agent::new(Config::default());
        let (ans, _) = a.run_with("hi", |_| String::new());
        assert_eq!(ans, "Stopped: empty response.");
    }

    #[test]
    fn loop_stops_at_final_answer() {
        let a = Agent::new(Config::default());
        let (ans, _) = a.run_with("hi", |_| "done, no call".to_string());
        assert_eq!(ans, "done, no call");
    }

    #[test]
    fn loop_guards_repeat() {
        let a = Agent::new(Config::default());
        let (ans, _) = a.run_with("hi", |_| {
            "[solve_symbolic(op=eval, expression=\"2+2\")]".to_string()
        });
        assert!(ans.contains("repeated"));
    }

    #[test]
    fn allowlist_blocks_unlisted_tool() {
        let cfg = Config {
            allowlist: vec!["solve_symbolic".into()],
            ..Config::default()
        };
        let a = Agent::new(cfg);
        let (ans, _) = a.run_with("hi", |msgs| {
            if msgs.iter().any(|m| m.role == "tool") {
                "final answer".to_string()
            } else {
                "[plot_chart(kind=function, title=\"t\")]".to_string()
            }
        });
        assert_eq!(ans, "final answer");
    }

    #[test]
    fn loop_dispatches_todo_write() {
        let a = Agent::new(Config::default());
        let mut n = 0;
        let (ans, _) = a.run_with("plan it", |msgs| {
            n += 1;
            if n == 1 {
                "[todo_write(op=create, content=\"step one\")]".to_string()
            } else {
                let tool = msgs.iter().find(|m| m.role == "tool").expect("tool msg");
                assert!(
                    tool.content.contains("Plan: [1 open, 0 done]"),
                    "{}",
                    tool.content
                );
                "done".to_string()
            }
        });
        assert_eq!(ans, "done");
    }

    #[test]
    fn two_step_solve_then_plot() {
        let a = Agent::new(Config::default());
        let mut n = 0;
        let (ans, _) = a.run_with("trip", |_| {
            n += 1;
            match n {
                1 => "[solve_symbolic(op=eval, expression=\"2+2\")]".to_string(),
                2 => "[plot_chart(kind=function, title=\"t\")]".to_string(),
                _ => "done: 4.".to_string(),
            }
        });
        assert!(ans.contains("done"));
    }

    #[test]
    fn clean_answer_strips_think() {
        assert_eq!(
            Agent::clean_answer("Hi <think>secret</think> there"),
            "Hi  there"
        );
    }
}
