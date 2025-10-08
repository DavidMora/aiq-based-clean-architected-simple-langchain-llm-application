"""Interfaces for LLM service layer."""
from abc import ABC, abstractmethod
from typing import AsyncIterator, List, Optional

from src.domain.entities.message import ChatMessage, ChatResponse


class ILLMService(ABC):
    """Interface for LLM service."""

    @abstractmethod
    async def chat(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None
    ) -> ChatResponse:
        """Send chat messages and get response.

        Args:
            messages: List of chat messages
            system_prompt: Optional system prompt

        Returns:
            ChatResponse with message and parsed output
        """
        pass

    @abstractmethod
    async def stream(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Stream chat response.

        Args:
            messages: List of chat messages
            system_prompt: Optional system prompt

        Yields:
            Chunks of the response text
        """
        pass


class IOutputParser(ABC):
    """Interface for output parser."""

    @abstractmethod
    def parse(self, text: str) -> dict:
        """Parse LLM output text.

        Args:
            text: Raw text output from LLM

        Returns:
            Parsed output as dictionary
        """
        pass
