"""LangChain adapter for LLM service."""
from typing import AsyncIterator, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.language_models import BaseChatModel

from src.domain.entities.message import ChatMessage, ChatResponse, MessageRole
from src.domain.interfaces.llm_service import ILLMService


class LangChainLLMAdapter(ILLMService):
    """Adapter for LangChain LLM models."""

    def __init__(self, llm: BaseChatModel):
        """Initialize LangChain adapter.

        Args:
            llm: LangChain chat model instance
        """
        self.llm = llm

    def _convert_messages(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None
    ) -> List:
        """Convert domain messages to LangChain messages.

        Args:
            messages: Domain chat messages
            system_prompt: Optional system prompt

        Returns:
            List of LangChain messages
        """
        lc_messages = []

        if system_prompt:
            lc_messages.append(SystemMessage(content=system_prompt))

        for msg in messages:
            if msg.role == MessageRole.USER:
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == MessageRole.ASSISTANT:
                lc_messages.append(AIMessage(content=msg.content))
            elif msg.role == MessageRole.SYSTEM:
                lc_messages.append(SystemMessage(content=msg.content))

        return lc_messages

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
            ChatResponse with message
        """
        lc_messages = self._convert_messages(messages, system_prompt)
        response = await self.llm.ainvoke(lc_messages)

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response.content
            )
        )

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
        lc_messages = self._convert_messages(messages, system_prompt)

        async for chunk in self.llm.astream(lc_messages):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content
