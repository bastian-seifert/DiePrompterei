"""
Abstract LLM client interface with provider implementations.

Supports Anthropic Claude and OpenAI GPT models.
"""

import json
import os
import re
from abc import ABC, abstractmethod

from .models import LLMConfig, LLMResponse


class LLMClient(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def generate(
        self, system_prompt: str, user_prompt: str, temperature: float
    ) -> LLMResponse:
        """
        Generate text response from LLM.

        Args:
            system_prompt: System/instruction prompt
            user_prompt: User message
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            LLMResponse with text and token counts
        """
        pass

    def generate_json(
        self, system_prompt: str, user_prompt: str, temperature: float
    ) -> dict:
        """
        Generate and parse JSON response from LLM.

        Args:
            system_prompt: System/instruction prompt
            user_prompt: User message
            temperature: Sampling temperature

        Returns:
            Parsed JSON dict

        Raises:
            ValueError: If response is not valid JSON
        """
        response = self.generate(system_prompt, user_prompt, temperature)
        return self._extract_json(response.text)

    def _extract_json(self, text: str) -> dict:
        """
        Extract JSON from LLM response.

        Handles both direct JSON and markdown code blocks.

        Args:
            text: Raw LLM response text

        Returns:
            Parsed JSON dict

        Raises:
            ValueError: If no valid JSON found
        """
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from ```json block
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try extracting from ``` block (without json marker)
        match = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not extract valid JSON from response: {text[:200]}...")


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""

    def __init__(self, model: str = "claude-sonnet-4-5"):
        """
        Initialize Anthropic client.

        Args:
            model: Model identifier

        Raises:
            ValueError: If ANTHROPIC_API_KEY not set
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Get your API key from https://console.anthropic.com/"
            )

        from anthropic import Anthropic

        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate(
        self, system_prompt: str, user_prompt: str, temperature: float
    ) -> LLMResponse:
        """Generate response using Claude API."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        return LLMResponse(
            text=response.content[0].text,
            tokens_input=response.usage.input_tokens,
            tokens_output=response.usage.output_tokens,
        )


class OpenAIClient(LLMClient):
    """OpenAI GPT API client."""

    def __init__(self, model: str = "gpt-4"):
        """
        Initialize OpenAI client.

        Args:
            model: Model identifier

        Raises:
            ValueError: If OPENAI_API_KEY not set
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Get your API key from https://platform.openai.com/"
            )

        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(
        self, system_prompt: str, user_prompt: str, temperature: float
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return LLMResponse(
            text=response.choices[0].message.content,
            tokens_input=response.usage.prompt_tokens,
            tokens_output=response.usage.completion_tokens,
        )


def create_llm_client(config: LLMConfig) -> LLMClient:
    """
    Factory function to create LLM client based on config.

    Args:
        config: LLM configuration

    Returns:
        Configured LLM client

    Raises:
        ValueError: If provider not supported
    """
    if config.provider == "anthropic":
        return AnthropicClient(model=config.model)
    elif config.provider == "openai":
        return OpenAIClient(model=config.model)
    else:
        raise ValueError(
            f"Unsupported LLM provider: {config.provider}. "
            f"Supported: anthropic, openai"
        )
