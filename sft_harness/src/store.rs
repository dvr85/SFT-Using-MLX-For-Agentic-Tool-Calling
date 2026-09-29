use redb::{Database, ReadableTable, TableDefinition};
use std::path::Path;

const TODOS: TableDefinition<u64, &str> = TableDefinition::new("todos");

/// Embedded per-session state backing the `todo_write` tool.
pub struct SessionStore {
    db: Database,
}

impl SessionStore {
    pub fn open(path: &Path) -> Result<Self, String> {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
        }
        let db: Database = Database::create(path).map_err(|e| e.to_string())?;
        let tx = db.begin_write().map_err(|e| e.to_string())?;
        tx.open_table(TODOS).map_err(|e| e.to_string())?;
        tx.commit().map_err(|e| e.to_string())?;
        Ok(Self { db })
    }

    fn items(&self) -> Result<Vec<(bool, String)>, String> {
        let tx = self.db.begin_read().map_err(|e| e.to_string())?;
        let t = tx.open_table(TODOS).map_err(|e| e.to_string())?;
        let mut out = vec![];
        for e in t.iter().map_err(|e| e.to_string())? {
            let (k, v) = e.map_err(|e| e.to_string())?;
            out.push((k.value() % 2 == 1, v.value().to_string()));
        }
        Ok(out)
    }

    fn echo(&self) -> String {
        let items = self.items().unwrap_or_default();
        let (mut open, mut done) = (0, 0);
        let mut lines = vec![];
        for (is_done, c) in &items {
            if *is_done {
                done += 1;
                lines.push(format!("- [x] {c}"));
            } else {
                open += 1;
                lines.push(format!("- [ ] {c}"));
            }
        }
        format!("Plan: [{open} open, {done} done]\n{}", lines.join("\n"))
    }

    /// `create` replaces the plan, `update` appends, `complete` marks all done.
    pub fn todo(&self, op: &str, content: &str) -> String {
        match op {
            "create" | "update" => {
                if content.is_empty() {
                    return "Invalid arguments: content required.".to_string();
                }
                let tx = match self.db.begin_write() {
                    Ok(t) => t,
                    Err(e) => return format!("Tool error: {e}."),
                };
                {
                    let mut t = match tx.open_table(TODOS) {
                        Ok(t) => t,
                        Err(e) => return format!("Tool error: {e}."),
                    };
                    if op == "create" {
                        // even keys = open, odd = done; wipe then insert one open item
                        let mut keys = vec![];
                        if let Ok(iter) = t.iter() {
                            for (k, _) in iter.flatten() {
                                keys.push(k.value());
                            }
                        }
                        for k in keys {
                            let _ = t.remove(k);
                        }
                        let _ = t.insert(0u64, content);
                    } else {
                        let mut max: Option<u64> = None;
                        if let Ok(iter) = t.iter() {
                            for (k, _) in iter.flatten() {
                                max = Some(max.map_or(k.value(), |m: u64| m.max(k.value())));
                            }
                        }
                        let _ = t.insert(max.map_or(0, |m| m + 2), content);
                    }
                }
                let result = match tx.commit() {
                    Ok(()) => self.echo(),
                    Err(e) => format!("Tool error: {e}."),
                };
                result
            }
            "complete" => {
                let tx = match self.db.begin_write() {
                    Ok(t) => t,
                    Err(e) => return format!("Tool error: {e}."),
                };
                {
                    let mut t = match tx.open_table(TODOS) {
                        Ok(t) => t,
                        Err(e) => return format!("Tool error: {e}."),
                    };
                    // flip even (open) keys to odd (done) preserving order
                    let mut rows: Vec<(u64, String)> = vec![];
                    if let Ok(iter) = t.iter() {
                        for (k, v) in iter.flatten() {
                            rows.push((k.value(), v.value().to_string()));
                        }
                    }
                    for (k, v) in rows {
                        if k % 2 == 0 {
                            let _ = t.remove(k);
                            let _ = t.insert(k + 1, v.as_str());
                        }
                    }
                }
                let result = match tx.commit() {
                    Ok(()) => self.echo(),
                    Err(e) => format!("Tool error: {e}."),
                };
                result
            }
            _ => "Invalid arguments: op must be create|update|complete.".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> std::path::PathBuf {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("sft-test-{}-{n}.redb", std::process::id()))
    }

    #[test]
    fn todo_lifecycle_echo() {
        let s = SessionStore::open(&tmp()).unwrap();
        let e = s.todo("create", "book flights");
        assert!(e.contains("[1 open, 0 done]") && e.contains("[ ] book flights"));
        let e = s.todo("update", "pack bags");
        assert!(e.contains("[2 open, 0 done]"));
        let e = s.todo("complete", "");
        assert!(e.contains("[0 open, 2 done]"));
    }

    #[test]
    fn todo_unknown_op() {
        let s = SessionStore::open(&tmp()).unwrap();
        assert_eq!(
            s.todo("delete", "x"),
            "Invalid arguments: op must be create|update|complete."
        );
    }

    #[test]
    fn todo_create_replaces_plan() {
        let s = SessionStore::open(&tmp()).unwrap();
        s.todo("create", "first");
        s.todo("update", "second");
        let e = s.todo("create", "fresh");
        assert!(e.contains("[1 open, 0 done]") && e.contains("fresh"));
        assert!(!e.contains("second"));
    }
}
