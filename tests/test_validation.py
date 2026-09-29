from src.data.loader import validate_conversation_format


class TestValidateConversationFormat:
    def test_valid_conversations(self):
        convs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        assert validate_conversation_format(convs) is True

    def test_missing_role(self):
        convs = [{"content": "Hello"}]
        assert validate_conversation_format(convs) is False

    def test_invalid_role(self):
        convs = [{"role": "invalid", "content": "Hello"}]
        assert validate_conversation_format(convs) is False

    def test_empty_content(self):
        convs = [{"role": "user", "content": ""}]
        assert validate_conversation_format(convs) is False
