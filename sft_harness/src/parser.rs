use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

fn name_paren_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"([A-Za-z_][A-Za-z0-9_]*)\(").unwrap())
}

fn bracket_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // ponytail: mirrors src/eval.py BRACKET_CALL_RE; prose like "[note]" has no paren so never matches
    RE.get_or_init(|| Regex::new(r"\[([^\[\]]+?)\(").unwrap())
}

fn xml_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?s)<tool_call>\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\{.*?\})?\s*</tool_call>")
            .unwrap()
    })
}

fn native_fn_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r#"(?s)<function\s+name="([A-Za-z_][A-Za-z0-9_]*)"\s*>(.*?)</function>"#)
            .unwrap()
    })
}

fn native_param_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r#"(?s)<param\s+name="([^"]+)"\s*>(.*?)</param>"#).unwrap())
}

fn strip_cdata(s: &str) -> String {
    let t = s.trim();
    if t.starts_with("<![CDATA[") && t.ends_with("]]>") {
        t[9..t.len() - 3].to_string()
    } else {
        t.to_string()
    }
}

/// Split a bracket arg string on top-level commas (respect nesting + quotes).
fn split_top_level(s: &str) -> Vec<String> {
    let mut parts = vec![];
    let (mut depth_c, mut depth_b, mut start) = (0i32, 0i32, 0usize);
    let (mut sq, mut dq) = (false, false);
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\'' && !dq {
            sq = !sq;
        } else if c == '"' && !sq {
            dq = !dq;
        } else if !sq && !dq {
            match c {
                '(' | '{' | '[' => {
                    if c == '(' {
                        depth_c += 1;
                    } else {
                        depth_b += 1;
                    }
                }
                ')' => depth_c -= 1,
                '}' | ']' => depth_b -= 1,
                ',' if depth_c == 0 && depth_b == 0 => {
                    parts.push(s[start..i].to_string());
                    start = i + 1;
                }
                _ => {}
            }
        }
        i += 1;
    }
    parts.push(s[start..].to_string());
    parts
}

fn parse_value(s: &str) -> serde_json::Value {
    let t = s.trim();
    // glm47: strings stay strings even if numeric-looking when quoted
    if (t.starts_with('\'') && t.ends_with('\'') && t.len() >= 2)
        || (t.starts_with('"') && t.ends_with('"') && t.len() >= 2)
    {
        let inner = &t[1..t.len() - 1];
        // qwen3_coder: single-quote dicts -> try JSON with quotes swapped
        if (inner.contains('\'') || inner.contains(':')) && inner.contains(':') {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(inner) {
                return v;
            }
        }
        return serde_json::Value::String(inner.to_string());
    }
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(t) {
        return v;
    }
    // single-quoted containers: swap to double quotes and retry
    if t.contains('\'') {
        let swapped: String = t.chars().map(|c| if c == '\'' { '"' } else { c }).collect();
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&swapped) {
            return v;
        }
    }
    serde_json::Value::String(t.to_string())
}

/// Extract the balanced `(...)` args starting at `open_idx` (index of `(`).
fn balanced_args(text: &str, open_idx: usize) -> Option<(String, usize)> {
    let b = text.as_bytes();
    let (mut sq, mut dq, mut depth) = (false, false, 0i32);
    let mut i = open_idx;
    while i < b.len() {
        let c = b[i] as char;
        if c == '\'' && !dq {
            sq = !sq;
        } else if c == '"' && !sq {
            dq = !dq;
        } else if !sq && !dq {
            if c == '(' {
                depth += 1;
            } else if c == ')' {
                depth -= 1;
                if depth == 0 {
                    return Some((text[open_idx + 1..i].to_string(), i + 1));
                }
            }
        }
        i += 1;
    }
    None
}

fn kwargs_of(args: &str) -> Result<serde_json::Value, String> {
    let mut map = serde_json::Map::new();
    if args.trim().is_empty() {
        return Ok(serde_json::Value::Object(map));
    }
    for part in split_top_level(args) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let eq = part
            .find('=')
            .ok_or_else(|| format!("Invalid arguments: bad arg '{part}'."))?;
        let (k, v) = (part[..eq].trim(), part[eq + 1..].trim());
        if k.is_empty() {
            return Err(format!("Invalid arguments: bad arg '{part}'."));
        }
        map.insert(k.to_string(), parse_value(v));
    }
    Ok(serde_json::Value::Object(map))
}

fn push_call(
    out: &mut Vec<ToolCall>,
    name: String,
    seg: &str,
    open_idx: usize,
) -> Result<(), String> {
    match balanced_args(seg, open_idx) {
        Some((args_str, _)) => out.push(ToolCall {
            name,
            arguments: kwargs_of(&args_str)?,
        }),
        None => return Err("Invalid arguments: unbalanced parens.".to_string()),
    }
    Ok(())
}
pub fn parse_tool_calls(text: &str) -> Result<Vec<ToolCall>, String> {
    // [TOOL_CALLS] split marker (mistral): parse each segment
    let segments: Vec<&str> = if text.contains("[TOOL_CALLS]") {
        text.split("[TOOL_CALLS]").collect()
    } else {
        vec![text]
    };
    let mut out = vec![];
    for seg in segments {
        // 1. MiniCPM native <function name=".."><param name="..">..</param></function>
        for cap in native_fn_re().captures_iter(seg) {
            let name = cap[1].to_string();
            let mut map = serde_json::Map::new();
            for p in native_param_re().captures_iter(&cap[2]) {
                map.insert(p[1].to_string(), parse_value(&strip_cdata(&p[2])));
            }
            out.push(ToolCall {
                name,
                arguments: serde_json::Value::Object(map),
            });
        }
        if !out.is_empty() {
            return Ok(out);
        }
        // 2. legacy <tool_call>name{json}</tool_call>
        for cap in xml_re().captures_iter(seg) {
            let name = cap[1].to_string();
            let args: serde_json::Value = match cap.get(2) {
                Some(m) => serde_json::from_str(m.as_str())
                    .map_err(|e| format!("Invalid arguments: {e}."))?,
                None => serde_json::Value::Object(Default::default()),
            };
            out.push(ToolCall {
                name,
                arguments: args,
            });
        }
        if !out.is_empty() {
            return Ok(out);
        }
        // bracket form: all name(...) inside outer [...] (parallel calls have no '[' per call)
        let mut found = false;
        if seg.contains('[') && seg.contains(']') {
            let name_re = name_paren_re();
            for m in name_re.find_iter(seg) {
                // skip names outside the outer brackets
                let open_br = seg.find('[').unwrap_or(0);
                if m.start() < open_br {
                    continue;
                }
                found = true;
                push_call(
                    &mut out,
                    m.as_str().trim_end_matches('(').to_string(),
                    seg,
                    m.end() - 1,
                )?;
            }
        } else {
            for m in bracket_re().find_iter(seg) {
                found = true;
                push_call(
                    &mut out,
                    seg[m.start() + 1..m.end() - 1].trim().to_string(),
                    seg,
                    m.end() - 1,
                )?;
            }
        }
        if found {
            return Ok(out);
        }
    }
    Err("No tool call found.".to_string())
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xml_first() {
        let calls =
            parse_tool_calls(r#"<tool_call>get_weather{"city": "Paris"}</tool_call>"#).unwrap();
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["city"], "Paris");
    }

    #[test]
    fn bracket_basic() {
        let calls = parse_tool_calls(r#"[get_weather(city="Paris")]"#).unwrap();
        assert_eq!(calls[0].name, "get_weather");
    }

    #[test]
    fn single_quote_comma() {
        let calls = parse_tool_calls("[search(query='hello, world')]").unwrap();
        assert_eq!(calls[0].arguments["query"], "hello, world");
    }

    #[test]
    fn parallel_calls() {
        let calls = parse_tool_calls("[multiply(a=2, b=3), search(query=\"x\")]").unwrap();
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn prose_is_err() {
        assert!(parse_tool_calls("[see note]").is_err());
        assert!(parse_tool_calls("just prose").is_err());
    }

    #[test]
    fn tool_calls_split_marker() {
        let calls =
            parse_tool_calls("[multiply(a=2, b=3)] [TOOL_CALLS] [search(query=\"x\")]").unwrap();
        assert_eq!(calls[0].name, "multiply");
    }

    #[test]
    fn mistral_trailing_text() {
        let calls = parse_tool_calls("[get_weather(city=\"Paris\")] extra prose here").unwrap();
        assert_eq!(calls[0].name, "get_weather");
    }

    #[test]
    fn qwen_single_quote_dict() {
        let calls = parse_tool_calls("[search(query={'k': 'v'})]").unwrap();
        assert_eq!(calls[0].arguments["query"]["k"], "v");
    }

    #[test]
    fn glm_string_stays_string() {
        let calls = parse_tool_calls("[multiply(a=\"2\", b=3)]").unwrap();
        assert_eq!(calls[0].arguments["a"], "2");
    }

    #[test]
    fn native_function_first() {
        let calls = parse_tool_calls(
            r#"<function name="search"><param name="query">Paris weather</param><param name="limit">3</param></function>"#,
        )
        .unwrap();
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].arguments["query"], "Paris weather");
    }

    #[test]
    fn native_function_cdata() {
        let calls = parse_tool_calls(
            r#"<function name="search"><param name="query"><![CDATA[a<b & c]]></param></function>"#,
        )
        .unwrap();
        assert_eq!(calls[0].arguments["query"], "a<b & c");
    }

    #[test]
    fn partial_call_is_err() {
        assert!(parse_tool_calls("<tool_call>multiply").is_err()); // partial -> Err
    }
}
