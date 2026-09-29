use serde::Serialize;

#[derive(Serialize)]
struct Event<'a> {
    source: &'a str,
    payload: &'a str,
    ms: u128,
}

/// JSONL trace to stdout only.
pub struct Tracer {
    t0: std::time::Instant,
}

impl Tracer {
    pub fn new() -> Self {
        Self {
            t0: std::time::Instant::now(),
        }
    }

    pub fn log(&mut self, source: &str, payload: &str) {
        let e = Event {
            source,
            payload,
            ms: self.t0.elapsed().as_millis(),
        };
        println!("{}", serde_json::to_string(&e).unwrap());
    }
}

impl Default for Tracer {
    fn default() -> Self {
        Self::new()
    }
}
