"""Use cases for chat operations."""
from typing import AsyncIterator, List, Optional

from src.domain.entities.message import ChatMessage, ChatResponse
from src.domain.interfaces.llm_service import ILLMService, IOutputParser


class ChatUseCase:
    """Use case for handling chat operations."""

    def __init__(
        self,
        llm_service: ILLMService,
        output_parser: Optional[IOutputParser] = None
    ):
        """Initialize chat use case.

        Args:
            llm_service: LLM service implementation
            output_parser: Optional output parser for structured responses
        """
        self.llm_service = llm_service
        self.output_parser = output_parser

    async def execute(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        parse_output: bool = True
    ) -> ChatResponse:
        """Execute chat use case.

        Args:
            messages: List of chat messages
            system_prompt: Optional system prompt
            parse_output: Whether to parse the output

        Returns:
            ChatResponse with message and optional parsed output
        """
        response = await self.llm_service.chat(
            messages=messages,
            system_prompt=system_prompt
        )

        if parse_output and self.output_parser:
            try:
                response.parsed_output = self.output_parser.parse(
                    response.message.content
                )
            except (ValueError, Exception) as e:
                # If parsing fails, just log and return unparsed response
                response.parsed_output = None

        return response


class StreamChatUseCase:
    """Use case for streaming chat operations."""

    def __init__(self, llm_service: ILLMService):
        """Initialize stream chat use case.

        Args:
            llm_service: LLM service implementation
        """
        self.llm_service = llm_service

    async def execute(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Execute stream chat use case.

        Args:
            messages: List of chat messages
            system_prompt: Optional system prompt

        Yields:
            Chunks of the response text
        """
        async for chunk in self.llm_service.stream(
            messages=messages,
            system_prompt=system_prompt
        ):
            yield chunk
