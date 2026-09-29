use clap::Parser;
use sft_harness::{agent::Agent, config, trace::Tracer};

#[derive(Parser)]
struct Cli {
    /// Prompt text; leading "sft-harness-mlx" keyword is accepted and stripped.
    #[arg(long, default_value = "Solve x**2 - 5*x + 6 = 0.")]
    prompt: String,
    #[arg(long)]
    model_url: Option<String>,
    /// Model id for the chat-completions body; must match the server's --model.
    #[arg(long)]
    model: Option<String>,
    #[arg(long)]
    config: Option<String>,
}

fn main() {
    let cli = Cli::parse();
    let mut cfg = config::load(cli.config.as_deref());
    if let Some(u) = cli.model_url {
        cfg.model_url = u;
    }
    if let Some(m) = cli.model {
        cfg.model = m;
    }
    let mut tracer = Tracer::new();
    let agent = Agent::new(cfg);
    let (answer, steps) = agent.run(&cli.prompt, &mut tracer);
    println!("{}", serde_json::json!({"answer": answer, "steps": steps}));
}
