"""
Configuration settings for QURE backend

Supports multiple LLM providers: Anthropic, OpenAI, Groq, HuggingFace, Ollama
"""

from enum import Enum
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application
    app_name: str = "QURE"
    app_version: str = "0.1.0"
    debug: bool = False

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://qure:qure@localhost:5432/qure",
        description="PostgreSQL connection string"
    )
    db_echo: bool = False

    # LLM Configuration
    llm_provider: LLMProvider = Field(
        default=LLMProvider.ANTHROPIC,
        description="Primary LLM provider"
    )

    # Anthropic
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    anthropic_max_tokens: int = 4096
    anthropic_temperature: float = 0.7

    # OpenAI
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4-turbo-preview"
    openai_max_tokens: int = 4096
    openai_temperature: float = 0.7

    # Groq
    groq_api_key: Optional[str] = None
    groq_model: str = "mixtral-8x7b-32768"
    groq_max_tokens: int = 4096
    groq_temperature: float = 0.7

    # HuggingFace
    huggingface_api_key: Optional[str] = None
    huggingface_model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    huggingface_max_tokens: int = 4096
    huggingface_temperature: float = 0.7

    # Ollama (local)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama2"
    ollama_max_tokens: int = 4096
    ollama_temperature: float = 0.7

    # Agent Configuration
    agent_max_retries: int = 3
    agent_timeout: int = 60
    agent_confidence_threshold: float = 0.7

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    def get_llm_config(self, provider: Optional[LLMProvider] = None) -> dict:
        """Get configuration for specified LLM provider"""
        provider = provider or self.llm_provider

        if provider == LLMProvider.ANTHROPIC:
            return {
                "api_key": self.anthropic_api_key,
                "model": self.anthropic_model,
                "max_tokens": self.anthropic_max_tokens,
                "temperature": self.anthropic_temperature,
            }
        elif provider == LLMProvider.OPENAI:
            return {
                "api_key": self.openai_api_key,
                "model": self.openai_model,
                "max_tokens": self.openai_max_tokens,
                "temperature": self.openai_temperature,
            }
        elif provider == LLMProvider.GROQ:
            return {
                "api_key": self.groq_api_key,
                "model": self.groq_model,
                "max_tokens": self.groq_max_tokens,
                "temperature": self.groq_temperature,
            }
        elif provider == LLMProvider.HUGGINGFACE:
            return {
                "api_key": self.huggingface_api_key,
                "model": self.huggingface_model,
                "max_tokens": self.huggingface_max_tokens,
                "temperature": self.huggingface_temperature,
            }
        elif provider == LLMProvider.OLLAMA:
            return {
                "base_url": self.ollama_base_url,
                "model": self.ollama_model,
                "max_tokens": self.ollama_max_tokens,
                "temperature": self.ollama_temperature,
            }
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


# Global settings instance
settings = Settings()
