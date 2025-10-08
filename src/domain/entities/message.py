"""Domain entities for chat messages."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MessageRole(Enum):
    """Message role enumeration."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """Chat message entity."""
    role: MessageRole
    content: str
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert message to dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            "metadata": self.metadata
        }


@dataclass
class ChatResponse:
    """Chat response entity."""
    message: ChatMessage
    parsed_output: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert response to dictionary."""
        return {
            "message": self.message.to_dict(),
            "parsed_output": self.parsed_output
        }
