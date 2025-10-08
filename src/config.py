"""Application configuration."""
from typing import Optional, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False

    # LLM Configuration
    llm_provider: str = "openai"  # openai, nvidia, anthropic, etc.
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.7
    llm_max_tokens: Optional[int] = None

    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None

    # NVIDIA NIM Configuration
    nvidia_api_key: Optional[str] = None
    nvidia_base_url: Optional[str] = "https://integrate.api.nvidia.com/v1"

    # Parser Configuration
    enable_output_parsing: bool = False
    parser_type: str = "json"  # json, langchain, custom

    @field_validator('llm_max_tokens', mode='before')
    @classmethod
    def parse_max_tokens(cls, v: Union[str, int, None]) -> Optional[int]:
        """Parse max_tokens, treating empty string as None."""
        if v == '' or v is None:
            return None
        return int(v)

    @field_validator('openai_base_url', 'nvidia_base_url', mode='before')
    @classmethod
    def parse_optional_url(cls, v: Optional[str]) -> Optional[str]:
        """Parse optional URL, treating empty string as None."""
        if v == '':
            return None
        return v

    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
