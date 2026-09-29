/// Minimal sft_harness.yaml loader; precedence: explicit arg > env > file default.
#[derive(Debug, Clone)]
pub struct Config {
    pub max_steps: usize,
    pub max_tokens_per_turn: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub enable_thinking: bool,
    pub model_url: String,
    /// Model id sent in the chat-completions body. Must match the server's
    /// --model argument; the server resolves it as a repo id (a placeholder 404s).
    pub model: String,
    pub request_timeout_s: u64,
    pub tool_timeout_s: u64,
    pub allowlist: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_steps: 4,
            max_tokens_per_turn: 1024,
            temperature: 0.9,
            top_p: 0.95,
            enable_thinking: true,
            model_url: std::env::var("SFT_HARNESS_MODEL_URL")
                .unwrap_or_else(|_| "http://localhost:8080/v1".to_string()),
            model: std::env::var("SFT_HARNESS_MODEL")
                .unwrap_or_else(|_| "models/sft-minicpm5-1b-adapters/export/fused".to_string()),
            request_timeout_s: 60,
            tool_timeout_s: 15,
            allowlist: vec![
                "solve_symbolic".into(),
                "plot_chart".into(),
                "todo_write".into(),
            ],
        }
    }
}

fn yaml_map(text: &str) -> std::collections::HashMap<String, String> {
    text.lines()
        .filter_map(|l| l.trim().split_once(':'))
        .map(|(k, v)| (k.trim().to_string(), v.trim().trim_matches('"').to_string()))
        .collect()
}

macro_rules! set_num {
    ($m:expr, $cfg:expr, $($k:literal => $f:ident),*) => {
        $(if let Some(v) = $m.get($k).and_then(|s| s.parse().ok()) { $cfg.$f = v; })*
    };
}

pub fn load(path: Option<&str>) -> Config {
    let mut cfg = Config::default();
    let text = path
        .and_then(|p| std::fs::read_to_string(p).ok())
        .unwrap_or_default();
    if text.is_empty() {
        return cfg;
    }
    let m = yaml_map(&text);
    set_num!(m, cfg, "max_steps" => max_steps, "max_tokens_per_turn" => max_tokens_per_turn, "temperature" => temperature, "top_p" => top_p, "request_timeout_s" => request_timeout_s, "tool_timeout_s" => tool_timeout_s);
    if let Some(v) = m
        .get("enable_thinking")
        .and_then(|s| s.parse::<bool>().ok())
    {
        cfg.enable_thinking = v;
    }
    if let Some(v) = m.get("model_url") {
        cfg.model_url = v.clone();
    }
    if let Some(v) = m.get("model") {
        cfg.model = v.clone();
    }
    // allowlist block: lines starting with "- " after "allowlist:"
    if let Some(idx) = text
        .lines()
        .position(|l| l.trim().starts_with("allowlist:"))
    {
        let items: Vec<String> = text
            .lines()
            .skip(idx + 1)
            .take_while(|l| l.trim().starts_with("- "))
            .map(|l| l.trim()[2..].trim().to_string())
            .collect();
        if !items.is_empty() {
            cfg.allowlist = items;
        }
    }
    // env overrides file (explicit CLI applied by caller afterwards)
    if let Ok(u) = std::env::var("SFT_HARNESS_MODEL_URL") {
        cfg.model_url = u;
    }
    if let Ok(m) = std::env::var("SFT_HARNESS_MODEL") {
        cfg.model = m;
    }
    if let Some(v) = std::env::var("SFT_HARNESS_TOOL_TIMEOUT_S")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
    {
        cfg.tool_timeout_s = v;
    }
    cfg
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_tmp(name: &str, contents: &str) -> String {
        let path =
            std::env::temp_dir().join(format!("sft-harness-test-{name}-{}", std::process::id()));
        std::fs::write(&path, contents).unwrap();
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn parses_model_id_without_confusing_model_url() {
        let path = write_tmp(
            "model-id",
            "model_url: \"http://localhost:8080/v1\"\nmodel: \"my-fused\"\n",
        );
        let cfg = load(Some(&path));
        std::fs::remove_file(&path).ok();
        // Guard against env overrides leaking into the assertion.
        if std::env::var("SFT_HARNESS_MODEL").is_ok()
            || std::env::var("SFT_HARNESS_MODEL_URL").is_ok()
        {
            return;
        }
        assert_eq!(cfg.model, "my-fused");
        assert_eq!(cfg.model_url, "http://localhost:8080/v1");
    }

    #[test]
    fn default_model_is_not_a_placeholder() {
        if std::env::var("SFT_HARNESS_MODEL").is_ok() {
            return;
        }
        assert_ne!(Config::default().model, "sft-harness");
    }
}
