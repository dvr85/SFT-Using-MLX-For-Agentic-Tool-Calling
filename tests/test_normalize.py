from src.data.normalize import normalize_conversations, validate_conversation_format


class TestNormalizeConversations:
    def test_from_value_aliases(self):
        convs = [
            {"from": "human", "value": "Hi"},
            {"from": "gpt", "value": "Hello"},
        ]
        assert normalize_conversations(convs) == [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]

    def test_role_passthrough(self):
        convs = [{"role": "tool", "content": "ok"}]
        assert normalize_conversations(convs) == [{"role": "tool", "content": "ok"}]


class TestValidateConversationFormat:
    def test_valid(self):
        assert validate_conversation_format([{"role": "user", "content": "Hello"}]) is True

    def test_invalid_role(self):
        assert validate_conversation_format([{"role": "invalid", "content": "x"}]) is False

    def test_empty_content(self):
        assert validate_conversation_format([{"role": "user", "content": " "}]) is False

    def test_empty_list(self):
        assert validate_conversation_format([]) is False
