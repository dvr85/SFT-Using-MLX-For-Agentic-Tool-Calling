use crate::parser::ToolCall;
use crate::registry;
use crate::store::SessionStore;

/// Allowlist-closed dispatch; errors become tool-message strings.
pub fn execute(
    call: &ToolCall,
    allowlist: &[String],
    store: &SessionStore,
    timeout_s: u64,
) -> String {
    if !allowlist.iter().any(|t| t == &call.name) {
        return format!(
            "Unknown tool: {}. Available: {}.",
            call.name,
            allowlist.join(", ")
        );
    }
    if !call.arguments.is_object() {
        return "Invalid arguments: expected JSON object.".to_string();
    }
    registry::dispatch(&call.name, &call.arguments, store, timeout_s)
}
