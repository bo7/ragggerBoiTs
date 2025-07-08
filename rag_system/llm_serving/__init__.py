"""
LLM Serving Module
==================

Provides LLM serving capabilities using OpenRouter API, vLLM, and Ollama backends.
Supports multiple open-source models via OpenRouter for immediate access,
with local deployment options for complete data sovereignty.
"""

from .openrouter_client import OpenRouterClient

# Import local servers conditionally
try:
    from .vllm_server import VLLMServer
    VLLM_AVAILABLE = True
except ImportError:
    VLLMServer = None
    VLLM_AVAILABLE = False

try:
    from .ollama_server import OllamaServer
    OLLAMA_AVAILABLE = True
except ImportError:
    OllamaServer = None
    OLLAMA_AVAILABLE = False

from .llm_client import LLMClient, LLMBackend

__all__ = [
    "OpenRouterClient",
    "LLMClient",
    "LLMBackend"
]

if VLLM_AVAILABLE:
    __all__.append("VLLMServer")

if OLLAMA_AVAILABLE:
    __all__.append("OllamaServer")