"""
LLM Service with multi-provider support

Provides a unified interface for:
- Anthropic Claude
- OpenAI GPT
- Groq
- HuggingFace
- Ollama (local)
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

import anthropic
import openai
import groq
from huggingface_hub import InferenceClient
import httpx

from backend.config.settings import Settings, LLMProvider


class BaseLLMProvider(ABC):
    """Base class for LLM providers"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate completion from prompt"""
        pass


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = anthropic.AsyncAnthropic(api_key=config["api_key"])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate completion using Claude"""
        messages = [{"role": "user", "content": prompt}]

        response = await self.client.messages.create(
            model=self.config["model"],
            max_tokens=kwargs.get("max_tokens", self.config["max_tokens"]),
            temperature=kwargs.get("temperature", self.config["temperature"]),
            system=system_prompt or "You are a helpful AI assistant specialized in financial reconciliation and compliance.",
            messages=messages
        )

        return response.content[0].text


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = openai.AsyncOpenAI(api_key=config["api_key"])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate completion using GPT"""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.config["model"],
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.config["max_tokens"]),
            temperature=kwargs.get("temperature", self.config["temperature"]),
        )

        return response.choices[0].message.content


class GroqProvider(BaseLLMProvider):
    """Groq provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = groq.AsyncGroq(api_key=config["api_key"])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate completion using Groq"""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.config["model"],
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.config["max_tokens"]),
            temperature=kwargs.get("temperature", self.config["temperature"]),
        )

        return response.choices[0].message.content


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace Inference API provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = InferenceClient(token=config["api_key"])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate completion using HuggingFace"""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"

        # Run sync client in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.text_generation(
                full_prompt,
                model=self.config["model"],
                max_new_tokens=kwargs.get("max_tokens", self.config["max_tokens"]),
                temperature=kwargs.get("temperature", self.config["temperature"]),
            )
        )

        return response


class OllamaProvider(BaseLLMProvider):
    """Ollama local model provider using direct HTTP calls"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config["base_url"]
        self.client = httpx.AsyncClient(timeout=60.0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate completion using Ollama"""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config["model"],
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config["temperature"]),
                "num_predict": kwargs.get("max_tokens", self.config["max_tokens"]),
            }
        }

        response = await self.client.post(
            f"{self.base_url}/api/chat",
            json=payload
        )
        response.raise_for_status()

        result = response.json()
        return result["message"]["content"]


class LLMService:
    """Unified LLM service supporting multiple providers"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._providers: Dict[LLMProvider, BaseLLMProvider] = {}

    def _get_provider_class(self, provider: LLMProvider):
        """Get provider class for given provider type"""
        provider_map = {
            LLMProvider.ANTHROPIC: AnthropicProvider,
            LLMProvider.OPENAI: OpenAIProvider,
            LLMProvider.GROQ: GroqProvider,
            LLMProvider.HUGGINGFACE: HuggingFaceProvider,
            LLMProvider.OLLAMA: OllamaProvider,
        }
        return provider_map[provider]

    def get_provider(self, provider: Optional[LLMProvider] = None) -> BaseLLMProvider:
        """Get or create provider instance"""
        provider = provider or self.settings.llm_provider

        if provider not in self._providers:
            config = self.settings.get_llm_config(provider)
            provider_class = self._get_provider_class(provider)
            self._providers[provider] = provider_class(config)

        return self._providers[provider]

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        provider: Optional[LLMProvider] = None,
        **kwargs
    ) -> str:
        """Generate completion using specified or default provider"""
        llm_provider = self.get_provider(provider)
        return await llm_provider.generate(prompt, system_prompt, **kwargs)

    async def generate_with_fallback(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        providers: Optional[List[LLMProvider]] = None,
        **kwargs
    ) -> str:
        """Try multiple providers in sequence until one succeeds"""
        if providers is None:
            providers = [self.settings.llm_provider]

        last_error = None
        for provider in providers:
            try:
                return await self.generate(prompt, system_prompt, provider, **kwargs)
            except Exception as e:
                last_error = e
                continue

        raise Exception(f"All providers failed. Last error: {last_error}")
