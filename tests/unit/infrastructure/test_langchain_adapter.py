"""Tests for LangChain adapter."""
import pytest
from typing import List

from src.domain.entities.message import ChatMessage, MessageRole
from src.infrastructure.llm.langchain_adapter import LangChainLLMAdapter


class TestLangChainLLMAdapter:
    """Tests for LangChainLLMAdapter."""

    @pytest.mark.asyncio
    async def test_chat(self, mock_langchain_llm):
        """Test chat method."""
        adapter = LangChainLLMAdapter(llm=mock_langchain_llm)

        messages = [
            ChatMessage(role=MessageRole.USER, content="Hello!")
        ]

        response = await adapter.chat(messages=messages)

        assert response.message.role == MessageRole.ASSISTANT
        assert response.message.content == "LangChain response"
        mock_langchain_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_with_system_prompt(self, mock_langchain_llm):
        """Test chat with system prompt."""
        adapter = LangChainLLMAdapter(llm=mock_langchain_llm)

        messages = [
            ChatMessage(role=MessageRole.USER, content="What's 2+2?")
        ]

        response = await adapter.chat(
            messages=messages,
            system_prompt="You are a helpful math assistant."
        )

        assert response.message.role == MessageRole.ASSISTANT
        # Verify system message was included
        call_args = mock_langchain_llm.ainvoke.call_args[0][0]
        assert len(call_args) == 2  # System + User message
        assert call_args[0].content == "You are a helpful math assistant."

    @pytest.mark.asyncio
    async def test_chat_with_multiple_messages(self, mock_langchain_llm):
        """Test chat with conversation history."""
        adapter = LangChainLLMAdapter(llm=mock_langchain_llm)

        messages = [
            ChatMessage(role=MessageRole.USER, content="Hello!"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"),
            ChatMessage(role=MessageRole.USER, content="How are you?")
        ]

        response = await adapter.chat(messages=messages)

        assert response.message.role == MessageRole.ASSISTANT
        call_args = mock_langchain_llm.ainvoke.call_args[0][0]
        assert len(call_args) == 3

    @pytest.mark.asyncio
    async def test_stream(self, mock_langchain_llm):
        """Test stream method."""
        adapter = LangChainLLMAdapter(llm=mock_langchain_llm)

        messages = [
            ChatMessage(role=MessageRole.USER, content="Tell me a story")
        ]

        chunks = []
        async for chunk in adapter.stream(messages=messages):
            chunks.append(chunk)

        assert chunks == ["Lang", "Chain", "stream"]

    @pytest.mark.asyncio
    async def test_stream_with_system_prompt(self, mock_langchain_llm):
        """Test stream with system prompt."""
        adapter = LangChainLLMAdapter(llm=mock_langchain_llm)

        messages = [
            ChatMessage(role=MessageRole.USER, content="Count to 3")
        ]

        chunks = []
        async for chunk in adapter.stream(
            messages=messages,
            system_prompt="You are a counting assistant."
        ):
            chunks.append(chunk)

        assert len(chunks) > 0

    def test_convert_messages_user(self):
        """Test converting user messages."""
        adapter = LangChainLLMAdapter(llm=None)

        messages = [
            ChatMessage(role=MessageRole.USER, content="Hello!")
        ]

        lc_messages = adapter._convert_messages(messages)

        assert len(lc_messages) == 1
        assert lc_messages[0].content == "Hello!"

    def test_convert_messages_with_system_prompt(self):
        """Test converting messages with system prompt."""
        adapter = LangChainLLMAdapter(llm=None)

        messages = [
            ChatMessage(role=MessageRole.USER, content="Hello!")
        ]

        lc_messages = adapter._convert_messages(
            messages,
            system_prompt="You are helpful."
        )

        assert len(lc_messages) == 2
        assert lc_messages[0].content == "You are helpful."
        assert lc_messages[1].content == "Hello!"
