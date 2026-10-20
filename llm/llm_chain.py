import os
import asyncio
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration settings for the LLM Chain - determines which implementation to use"""
    # LLM configuration - supports both Ollama and OpenAI-compatible APIs
    llm_base_url: Optional[str] = os.environ.get("LLM_BASE_URL")
    llm_model: Optional[str] = os.environ.get("LLM_MODEL")

    @property
    def use_custom_llm(self) -> bool:
        """Check if custom LLM (OpenAI-compatible) should be used instead of Ollama"""
        return self.llm_base_url is not None and self.llm_model is not None


class LLMChain:
    """
    Factory class that creates and delegates to either OllamaChain or OpenAIChain
    based on environment configuration.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the LLM Chain with optional custom configuration.
        Automatically selects the appropriate implementation (Ollama or OpenAI).

        Args:
            config: Optional LLMConfig to determine which chain to use
        """
        self.config = config or LLMConfig()

        # Select and initialize the appropriate chain implementation
        if self.config.use_custom_llm:
            logger.info("Initializing OpenAI-compatible LLM Chain")
            from openai_chain import OpenAIChain
            self.chain = OpenAIChain()
        else:
            logger.info("Initializing Ollama LLM Chain")
            from ollama_chain import OllamaChain
            self.chain = OllamaChain()

    def invoke(self, message: str, user_id: str = "default_user") -> dict:
        """
        Process a user message and return the response.
        Delegates to the underlying chain implementation.

        Args:
            message: The user's input message
            user_id: The user ID for filtering documents

        Returns:
            dict: Contains the model's response or error information
                  Format: {"answer": str, "links": set} or {"error": str, "status": str}
        """
        return self.chain.invoke(message, user_id=user_id)

    async def ainvoke(self, message: str, user_id: str = "default_user") -> dict:
        """Use the async custom-LLM graph so web requests remain cancellable."""
        if self.config.use_custom_llm:
            return await self.chain.ainvoke(message, user_id=user_id)
        return await asyncio.to_thread(self.chain.invoke, message, user_id=user_id)
