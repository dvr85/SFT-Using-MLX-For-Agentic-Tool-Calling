"""Conversation normalize/validate helpers."""

from typing import Any

VALID_ROLES = {"system", "user", "assistant", "tool"}

FROM_ALIASES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "assistant",
    "function_call": "assistant",
    "observation": "tool",
}


def normalize_conversations(conversations: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Accept {from,value} or {role,content} turns; return [{role,content}]."""
    normalized: list[dict[str, str]] = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        if "from" in turn or "value" in turn:
            raw_role = str(turn.get("from", "user"))
            content = str(turn.get("value", ""))
        else:
            raw_role = str(turn.get("role", "user"))
            content = str(turn.get("content", ""))
        role = FROM_ALIASES.get(raw_role.lower(), raw_role.lower())
        normalized.append({"role": role, "content": content})
    return normalized


def validate_conversation_format(conversations: list[dict[str, str]]) -> bool:
    """True if non-empty, valid roles, non-empty content."""
    if not conversations:
        return False
    for m in conversations:
        role = m.get("role", "")
        content = m.get("content", "")
        if role not in VALID_ROLES:
            return False
        if not content or not content.strip():
            return False
    return True
