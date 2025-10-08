"""API request/response schemas."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class MessageSchema(BaseModel):
    """Schema for chat message."""
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


class ChatRequest(BaseModel):
    """Schema for chat request."""
    messages: List[MessageSchema] = Field(..., description="List of chat messages")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt")
    parse_output: bool = Field(False, description="Whether to parse the output as JSON")


class ChatResponseSchema(BaseModel):
    """Schema for chat response."""
    message: MessageSchema
    parsed_output: Optional[Dict[str, Any]] = None


class StreamRequest(BaseModel):
    """Schema for stream request."""
    messages: List[MessageSchema] = Field(..., description="List of chat messages")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt")
