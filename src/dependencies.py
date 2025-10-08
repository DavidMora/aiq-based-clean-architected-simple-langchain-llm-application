"""Dependency injection setup."""
from functools import lru_cache
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from src.application.use_cases.chat_use_case import ChatUseCase, StreamChatUseCase
from src.domain.interfaces.llm_service import ILLMService, IOutputParser
from src.infrastructure.llm.langchain_adapter import LangChainLLMAdapter
from src.infrastructure.parsers.json_parser import JSONOutputParser
from src.config import settings


@lru_cache()
def get_llm_service() -> ILLMService:
    """Get LLM service instance.

    Returns:
        LLM service implementation
    """
    if settings.llm_provider == "openai":
        llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url
        )
    elif settings.llm_provider == "nvidia":
        llm = ChatNVIDIA(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

    return LangChainLLMAdapter(llm=llm)


@lru_cache()
def get_output_parser() -> Optional[IOutputParser]:
    """Get output parser instance.

    Returns:
        Output parser implementation or None
    """
    if not settings.enable_output_parsing:
        return None

    if settings.parser_type == "json":
        return JSONOutputParser(strict=False)

    return None


def get_chat_use_case() -> ChatUseCase:
    """Get chat use case instance.

    Returns:
        Chat use case
    """
    return ChatUseCase(
        llm_service=get_llm_service(),
        output_parser=get_output_parser()
    )


def get_stream_use_case() -> StreamChatUseCase:
    """Get stream chat use case instance.

    Returns:
        Stream chat use case
    """
    return StreamChatUseCase(
        llm_service=get_llm_service()
    )
