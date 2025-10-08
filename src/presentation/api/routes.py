"""API routes for chat endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncIterator

from src.application.use_cases.chat_use_case import ChatUseCase, StreamChatUseCase
from src.domain.entities.message import ChatMessage, MessageRole
from src.presentation.api.schemas import (
    ChatRequest,
    ChatResponseSchema,
    StreamRequest,
    MessageSchema
)
from src.dependencies import get_chat_use_case, get_stream_use_case


router = APIRouter(prefix="/api/v1", tags=["chat"])


def _convert_to_domain_messages(messages: list[MessageSchema]) -> list[ChatMessage]:
    """Convert API messages to domain messages.

    Raises:
        HTTPException: If role is invalid
    """
    try:
        return [
            ChatMessage(
                role=MessageRole(msg.role),
                content=msg.content,
                metadata=msg.metadata
            )
            for msg in messages
        ]
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid message role: {str(e)}"
        )


@router.post("/chat", response_model=ChatResponseSchema)
async def chat(
    request: ChatRequest,
    chat_use_case: ChatUseCase = Depends(get_chat_use_case)
) -> ChatResponseSchema:
    """Handle chat request.

    Args:
        request: Chat request with messages
        chat_use_case: Injected chat use case

    Returns:
        Chat response with message and optional parsed output
    """
    domain_messages = _convert_to_domain_messages(request.messages)

    response = await chat_use_case.execute(
        messages=domain_messages,
        system_prompt=request.system_prompt,
        parse_output=request.parse_output
    )

    return ChatResponseSchema(
        message=MessageSchema(
            role=response.message.role.value,
            content=response.message.content,
            metadata=response.message.metadata
        ),
        parsed_output=response.parsed_output
    )


@router.post("/chat/stream")
async def stream_chat(
    request: StreamRequest,
    stream_use_case: StreamChatUseCase = Depends(get_stream_use_case)
) -> StreamingResponse:
    """Handle streaming chat request.

    Args:
        request: Stream request with messages
        stream_use_case: Injected stream use case

    Returns:
        Streaming response
    """
    domain_messages = _convert_to_domain_messages(request.messages)

    async def generate() -> AsyncIterator[str]:
        """Generate streaming response."""
        async for chunk in stream_use_case.execute(
            messages=domain_messages,
            system_prompt=request.system_prompt
        ):
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
