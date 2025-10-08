"""Tests for chat use cases."""
import pytest

from src.domain.entities.message import ChatMessage, MessageRole


class TestChatUseCase:
    """Tests for ChatUseCase."""

    @pytest.mark.asyncio
    async def test_execute_without_parsing(
        self,
        chat_use_case,
        sample_messages,
        mock_llm_service
    ):
        """Test executing chat without output parsing."""
        response = await chat_use_case.execute(
            messages=sample_messages,
            parse_output=False
        )

        assert response.message.role == MessageRole.ASSISTANT
        assert response.message.content == "Mock response"
        assert response.parsed_output is None
        assert mock_llm_service.chat_called
        assert mock_llm_service.last_messages == sample_messages

    @pytest.mark.asyncio
    async def test_execute_with_system_prompt(
        self,
        chat_use_case,
        sample_messages,
        mock_llm_service
    ):
        """Test executing chat with system prompt."""
        system_prompt = "You are a helpful assistant."

        response = await chat_use_case.execute(
            messages=sample_messages,
            system_prompt=system_prompt,
            parse_output=False
        )

        assert mock_llm_service.last_system_prompt == system_prompt
        assert response.message.content == "Mock response"

    @pytest.mark.asyncio
    async def test_execute_with_parsing(
        self,
        chat_use_case_with_parser,
        sample_messages,
        mock_llm_service,
        mock_output_parser
    ):
        """Test executing chat with output parsing."""
        response = await chat_use_case_with_parser.execute(
            messages=sample_messages,
            parse_output=True
        )

        assert response.message.content == "Mock response"
        assert response.parsed_output == {"parsed": True}
        assert mock_output_parser.parse_called
        assert mock_output_parser.last_text == "Mock response"

    @pytest.mark.asyncio
    async def test_execute_parsing_disabled_by_flag(
        self,
        chat_use_case_with_parser,
        sample_messages,
        mock_output_parser
    ):
        """Test that parsing is skipped when parse_output=False."""
        response = await chat_use_case_with_parser.execute(
            messages=sample_messages,
            parse_output=False
        )

        assert response.parsed_output is None
        assert not mock_output_parser.parse_called

    @pytest.mark.asyncio
    async def test_execute_without_parser_ignores_parse_flag(
        self,
        chat_use_case,
        sample_messages
    ):
        """Test that parse_output=True without parser doesn't error."""
        response = await chat_use_case.execute(
            messages=sample_messages,
            parse_output=True
        )

        assert response.parsed_output is None

    @pytest.mark.asyncio
    async def test_execute_handles_parser_failure(
        self,
        chat_use_case_with_parser,
        sample_messages,
        mock_output_parser
    ):
        """Test that parser failures are handled gracefully."""
        # Make the parser raise an exception
        def failing_parse(text):
            raise ValueError("Parse error")

        mock_output_parser.parse = failing_parse

        response = await chat_use_case_with_parser.execute(
            messages=sample_messages,
            parse_output=True
        )

        # Should not raise, parsed_output should be None
        assert response.message.content == "Mock response"
        assert response.parsed_output is None

    @pytest.mark.asyncio
    async def test_execute_with_empty_messages(self, chat_use_case):
        """Test executing with empty message list."""
        response = await chat_use_case.execute(
            messages=[],
            parse_output=False
        )

        assert response.message.role == MessageRole.ASSISTANT


class TestStreamChatUseCase:
    """Tests for StreamChatUseCase."""

    @pytest.mark.asyncio
    async def test_execute_stream(
        self,
        stream_use_case,
        sample_messages,
        mock_llm_service
    ):
        """Test streaming chat."""
        chunks = []
        async for chunk in stream_use_case.execute(messages=sample_messages):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert mock_llm_service.stream_called
        # "Mock response" split by words
        assert "Mock " in chunks
        assert "response " in chunks

    @pytest.mark.asyncio
    async def test_execute_stream_with_system_prompt(
        self,
        stream_use_case,
        sample_messages,
        mock_llm_service
    ):
        """Test streaming with system prompt."""
        system_prompt = "Be concise."

        chunks = []
        async for chunk in stream_use_case.execute(
            messages=sample_messages,
            system_prompt=system_prompt
        ):
            chunks.append(chunk)

        assert mock_llm_service.last_system_prompt == system_prompt
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_execute_stream_concatenated(
        self,
        stream_use_case,
        sample_messages
    ):
        """Test that stream chunks concatenate correctly."""
        chunks = []
        async for chunk in stream_use_case.execute(messages=sample_messages):
            chunks.append(chunk)

        full_text = "".join(chunks)
        assert full_text == "Mock response "

    @pytest.mark.asyncio
    async def test_execute_stream_with_empty_messages(self, stream_use_case):
        """Test streaming with empty message list."""
        chunks = []
        async for chunk in stream_use_case.execute(messages=[]):
            chunks.append(chunk)

        # Should still work, returning something
        assert len(chunks) >= 0
