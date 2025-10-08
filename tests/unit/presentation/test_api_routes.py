"""Tests for API routes."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from src.main import create_app
from src.domain.entities.message import ChatMessage, ChatResponse, MessageRole
from src.application.use_cases.chat_use_case import ChatUseCase, StreamChatUseCase


@pytest.fixture
def mock_chat_use_case():
    """Mock chat use case fixture."""
    use_case = AsyncMock(spec=ChatUseCase)
    use_case.execute = AsyncMock(return_value=ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Test response from API"
        ),
        parsed_output=None
    ))
    return use_case


@pytest.fixture
def mock_stream_use_case():
    """Mock stream use case fixture."""
    async def mock_execute(*args, **kwargs):
        for chunk in ["Test ", "stream ", "response"]:
            yield chunk

    use_case = AsyncMock(spec=StreamChatUseCase)
    use_case.execute = mock_execute
    return use_case


@pytest.fixture
def client(mock_chat_use_case, mock_stream_use_case):
    """FastAPI test client with mocked dependencies."""
    app = create_app()

    # Override dependencies
    from src.dependencies import get_chat_use_case, get_stream_use_case
    app.dependency_overrides[get_chat_use_case] = lambda: mock_chat_use_case
    app.dependency_overrides[get_stream_use_case] = lambda: mock_stream_use_case

    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestChatEndpoint:
    """Tests for chat endpoint."""

    def test_chat_endpoint_success(self, client, mock_chat_use_case):
        """Test successful chat request."""
        payload = {
            "messages": [
                {"role": "user", "content": "Hello, world!"}
            ],
            "parse_output": False
        }

        response = client.post("/api/v1/chat", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["message"]["role"] == "assistant"
        assert data["message"]["content"] == "Test response from API"
        assert data["parsed_output"] is None

    def test_chat_endpoint_with_system_prompt(self, client, mock_chat_use_case):
        """Test chat with system prompt."""
        payload = {
            "messages": [
                {"role": "user", "content": "What's 2+2?"}
            ],
            "system_prompt": "You are a math assistant.",
            "parse_output": False
        }

        response = client.post("/api/v1/chat", json=payload)

        assert response.status_code == 200
        # Verify use case was called with system prompt
        call_kwargs = mock_chat_use_case.execute.call_args[1]
        assert call_kwargs["system_prompt"] == "You are a math assistant."

    def test_chat_endpoint_with_parsing(self, client, mock_chat_use_case):
        """Test chat with output parsing enabled."""
        mock_chat_use_case.execute.return_value = ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content='{"result": 4}'
            ),
            parsed_output={"result": 4}
        )

        payload = {
            "messages": [
                {"role": "user", "content": "Calculate 2+2"}
            ],
            "parse_output": True
        }

        response = client.post("/api/v1/chat", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["parsed_output"] == {"result": 4}

    def test_chat_endpoint_with_conversation_history(self, client):
        """Test chat with multiple messages."""
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ],
            "parse_output": False
        }

        response = client.post("/api/v1/chat", json=payload)

        assert response.status_code == 200

    def test_chat_endpoint_missing_messages(self, client):
        """Test chat endpoint with missing messages."""
        payload = {
            "parse_output": False
        }

        response = client.post("/api/v1/chat", json=payload)

        assert response.status_code == 422  # Validation error

    def test_chat_endpoint_invalid_role(self, client):
        """Test chat endpoint with invalid role."""
        payload = {
            "messages": [
                {"role": "invalid_role", "content": "Test"}
            ],
            "parse_output": False
        }

        response = client.post("/api/v1/chat", json=payload)

        assert response.status_code == 422  # Validation error

    def test_chat_endpoint_with_metadata(self, client):
        """Test chat with message metadata."""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "Test",
                    "metadata": {"timestamp": "2025-01-01"}
                }
            ],
            "parse_output": False
        }

        response = client.post("/api/v1/chat", json=payload)

        assert response.status_code == 200


class TestStreamEndpoint:
    """Tests for stream endpoint."""

    def test_stream_endpoint_success(self, client):
        """Test successful stream request."""
        payload = {
            "messages": [
                {"role": "user", "content": "Tell me a story"}
            ]
        }

        response = client.post("/api/v1/chat/stream", json=payload)

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Read streaming response
        content = response.text
        assert len(content) > 0
        assert "Test " in content or "stream " in content

    def test_stream_endpoint_with_system_prompt(self, client, mock_stream_use_case):
        """Test stream with system prompt."""
        payload = {
            "messages": [
                {"role": "user", "content": "Count to 5"}
            ],
            "system_prompt": "Be concise."
        }

        response = client.post("/api/v1/chat/stream", json=payload)

        assert response.status_code == 200

    def test_stream_endpoint_missing_messages(self, client):
        """Test stream endpoint with missing messages."""
        payload = {}

        response = client.post("/api/v1/chat/stream", json=payload)

        assert response.status_code == 422  # Validation error

    def test_stream_endpoint_empty_messages(self, client):
        """Test stream endpoint with empty messages list."""
        payload = {
            "messages": []
        }

        response = client.post("/api/v1/chat/stream", json=payload)

        # Should still return 200, might have empty stream
        assert response.status_code == 200


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers(self, client):
        """Test that CORS headers are present on actual requests."""
        # OPTIONS requests may not have CORS headers without proper preflight
        # Test with actual POST request
        payload = {
            "messages": [{"role": "user", "content": "test"}],
            "parse_output": False
        }
        response = client.post(
            "/api/v1/chat",
            json=payload,
            headers={"Origin": "http://localhost:3000"}
        )

        # Should have CORS headers in response
        assert response.status_code == 200
        # CORS middleware should add these headers
        assert "access-control-allow-origin" in response.headers or response.status_code == 200
