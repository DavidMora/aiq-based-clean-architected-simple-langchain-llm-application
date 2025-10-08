"""Tests for domain entities."""
import pytest

from src.domain.entities.message import ChatMessage, ChatResponse, MessageRole


class TestMessageRole:
    """Tests for MessageRole enum."""

    def test_message_role_values(self):
        """Test MessageRole enum values."""
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.SYSTEM.value == "system"


class TestChatMessage:
    """Tests for ChatMessage entity."""

    def test_create_chat_message(self):
        """Test creating a chat message."""
        message = ChatMessage(
            role=MessageRole.USER,
            content="Hello, world!",
            metadata={"timestamp": "2025-01-01"}
        )

        assert message.role == MessageRole.USER
        assert message.content == "Hello, world!"
        assert message.metadata == {"timestamp": "2025-01-01"}

    def test_create_chat_message_without_metadata(self):
        """Test creating a chat message without metadata."""
        message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Hi there!"
        )

        assert message.role == MessageRole.ASSISTANT
        assert message.content == "Hi there!"
        assert message.metadata is None

    def test_chat_message_to_dict(self):
        """Test converting chat message to dictionary."""
        message = ChatMessage(
            role=MessageRole.SYSTEM,
            content="System message",
            metadata={"priority": "high"}
        )

        result = message.to_dict()

        assert result == {
            "role": "system",
            "content": "System message",
            "metadata": {"priority": "high"}
        }

    def test_chat_message_to_dict_without_metadata(self):
        """Test converting chat message without metadata to dict."""
        message = ChatMessage(
            role=MessageRole.USER,
            content="Test"
        )

        result = message.to_dict()

        assert result == {
            "role": "user",
            "content": "Test",
            "metadata": None
        }


class TestChatResponse:
    """Tests for ChatResponse entity."""

    def test_create_chat_response(self):
        """Test creating a chat response."""
        message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Response content"
        )
        response = ChatResponse(
            message=message,
            parsed_output={"key": "value"}
        )

        assert response.message == message
        assert response.parsed_output == {"key": "value"}

    def test_create_chat_response_without_parsed_output(self):
        """Test creating a chat response without parsed output."""
        message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Simple response"
        )
        response = ChatResponse(message=message)

        assert response.message == message
        assert response.parsed_output is None

    def test_chat_response_to_dict(self):
        """Test converting chat response to dictionary."""
        message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Test response"
        )
        response = ChatResponse(
            message=message,
            parsed_output={"result": 42}
        )

        result = response.to_dict()

        assert result == {
            "message": {
                "role": "assistant",
                "content": "Test response",
                "metadata": None
            },
            "parsed_output": {"result": 42}
        }

    def test_chat_response_to_dict_without_parsed_output(self):
        """Test converting chat response without parsed output to dict."""
        message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Test"
        )
        response = ChatResponse(message=message)

        result = response.to_dict()

        assert result == {
            "message": {
                "role": "assistant",
                "content": "Test",
                "metadata": None
            },
            "parsed_output": None
        }
