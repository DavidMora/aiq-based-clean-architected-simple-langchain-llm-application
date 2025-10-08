"""Pytest configuration and fixtures."""
import pytest
from typing import AsyncIterator, List
from unittest.mock import AsyncMock, Mock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from src.domain.entities.message import ChatMessage, ChatResponse, MessageRole
from src.domain.interfaces.llm_service import ILLMService, IOutputParser
from src.infrastructure.llm.langchain_adapter import LangChainLLMAdapter
from src.infrastructure.parsers.json_parser import JSONOutputParser
from src.application.use_cases.chat_use_case import ChatUseCase, StreamChatUseCase


@pytest.fixture
def sample_messages() -> List[ChatMessage]:
    """Sample chat messages fixture."""
    return [
        ChatMessage(role=MessageRole.USER, content="Hello, how are you?"),
        ChatMessage(role=MessageRole.ASSISTANT, content="I'm doing well, thank you!"),
        ChatMessage(role=MessageRole.USER, content="What's 2+2?"),
    ]


@pytest.fixture
def sample_chat_response() -> ChatResponse:
    """Sample chat response fixture."""
    return ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content="The answer is 4."
        )
    )


@pytest.fixture
def sample_json_response() -> ChatResponse:
    """Sample JSON chat response fixture."""
    return ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content='{"result": 4, "explanation": "2+2 equals 4"}'
        )
    )


class MockLLMService(ILLMService):
    """Mock LLM service for testing."""

    def __init__(self, response_content: str = "Mock response"):
        self.response_content = response_content
        self.chat_called = False
        self.stream_called = False
        self.last_messages = None
        self.last_system_prompt = None

    async def chat(
        self,
        messages: List[ChatMessage],
        system_prompt: str = None
    ) -> ChatResponse:
        """Mock chat method."""
        self.chat_called = True
        self.last_messages = messages
        self.last_system_prompt = system_prompt
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=self.response_content
            )
        )

    async def stream(
        self,
        messages: List[ChatMessage],
        system_prompt: str = None
    ) -> AsyncIterator[str]:
        """Mock stream method."""
        self.stream_called = True
        self.last_messages = messages
        self.last_system_prompt = system_prompt
        for chunk in self.response_content.split():
            yield chunk + " "


class MockOutputParser(IOutputParser):
    """Mock output parser for testing."""

    def __init__(self, parsed_result: dict = None):
        self.parsed_result = parsed_result or {"parsed": True}
        self.parse_called = False
        self.last_text = None

    def parse(self, text: str) -> dict:
        """Mock parse method."""
        self.parse_called = True
        self.last_text = text
        return self.parsed_result


@pytest.fixture
def mock_llm_service() -> MockLLMService:
    """Mock LLM service fixture."""
    return MockLLMService()


@pytest.fixture
def mock_output_parser() -> MockOutputParser:
    """Mock output parser fixture."""
    return MockOutputParser()


@pytest.fixture
def mock_langchain_llm():
    """Mock LangChain LLM fixture."""
    mock_llm = AsyncMock(spec=BaseChatModel)
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="LangChain response"))

    async def mock_astream(messages):
        for chunk in ["Lang", "Chain", "stream"]:
            yield AIMessage(content=chunk)

    mock_llm.astream = mock_astream
    return mock_llm


@pytest.fixture
def chat_use_case(mock_llm_service: MockLLMService) -> ChatUseCase:
    """Chat use case fixture."""
    return ChatUseCase(llm_service=mock_llm_service)


@pytest.fixture
def chat_use_case_with_parser(
    mock_llm_service: MockLLMService,
    mock_output_parser: MockOutputParser
) -> ChatUseCase:
    """Chat use case with parser fixture."""
    return ChatUseCase(
        llm_service=mock_llm_service,
        output_parser=mock_output_parser
    )


@pytest.fixture
def stream_use_case(mock_llm_service: MockLLMService) -> StreamChatUseCase:
    """Stream use case fixture."""
    return StreamChatUseCase(llm_service=mock_llm_service)


@pytest.fixture
def json_parser() -> JSONOutputParser:
    """JSON parser fixture."""
    return JSONOutputParser(strict=False)
