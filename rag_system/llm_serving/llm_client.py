"""
LLM Client Interface
===================

Unified interface for interacting with different LLM backends (OpenRouter, vLLM, Ollama).
Provides automatic fallback and load balancing capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, AsyncGenerator
from enum import Enum

from .openrouter_client import OpenRouterClient
from ..config import get_settings

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

logger = logging.getLogger(__name__)


class LLMBackend(Enum):
    """Supported LLM backends"""
    OPENROUTER = "openrouter"
    VLLM = "vllm"
    OLLAMA = "ollama"


class LLMClient:
    """Unified LLM client with automatic fallback"""
    
    def __init__(self, preferred_backend: LLMBackend = LLMBackend.OPENROUTER):
        self.settings = get_settings()
        self.preferred_backend = preferred_backend
        self.current_backend: Optional[LLMBackend] = None
        
        # Initialize backends
        self.openrouter_client: Optional[OpenRouterClient] = None
        self.vllm_server: Optional[VLLMServer] = None
        self.ollama_server: Optional[OllamaServer] = None
        
        # State tracking
        self.is_initialized = False
        self.fallback_enabled = True
        
    async def initialize(self, model_name: str = None) -> bool:
        """Initialize the LLM client with the preferred backend"""
        try:
            model_name = model_name or self.settings.llm.model_name
            
            if self.preferred_backend == LLMBackend.OPENROUTER:
                success = await self._initialize_openrouter(model_name)
                if success:
                    self.current_backend = LLMBackend.OPENROUTER
                    self.is_initialized = True
                    return True
                elif self.fallback_enabled:
                    logger.warning("OpenRouter initialization failed, falling back to Ollama")
                    return await self._initialize_ollama_fallback()
            
            elif self.preferred_backend == LLMBackend.VLLM:
                success = await self._initialize_vllm(model_name)
                if success:
                    self.current_backend = LLMBackend.VLLM
                    self.is_initialized = True
                    return True
                elif self.fallback_enabled:
                    logger.warning("vLLM initialization failed, falling back to OpenRouter")
                    return await self._initialize_openrouter_fallback()
            
            elif self.preferred_backend == LLMBackend.OLLAMA:
                success = await self._initialize_ollama(model_name)
                if success:
                    self.current_backend = LLMBackend.OLLAMA
                    self.is_initialized = True
                    return True
                elif self.fallback_enabled:
                    logger.warning("Ollama initialization failed, falling back to OpenRouter")
                    return await self._initialize_openrouter_fallback()
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            return False
    
    async def _initialize_openrouter(self, model_name: str) -> bool:
        """Initialize OpenRouter backend"""
        try:
            # Map model name to OpenRouter format
            openrouter_model = self._map_to_openrouter_model(model_name)
            
            self.openrouter_client = OpenRouterClient()
            
            # Switch to the desired model
            if not self.openrouter_client.switch_model(openrouter_model):
                logger.error(f"Failed to switch to model: {openrouter_model}")
                return False
            
            # Test the connection
            if await self.openrouter_client.health_check():
                logger.info(f"OpenRouter initialized with model: {self.openrouter_client.model_name}")
                return True
            else:
                logger.error("OpenRouter health check failed")
                return False
                
        except Exception as e:
            logger.error(f"OpenRouter initialization failed: {e}")
            return False
    
    def _map_to_openrouter_model(self, model_name: str) -> str:
        """Map standard model names to OpenRouter format"""
        mapping = {
            "meta-llama/Llama-3.2-7B-Instruct": "llama-3.3-70b",
            "meta-llama/Llama-3.2-70B-Instruct": "llama-3.3-70b", 
            "mistralai/Mistral-7B-Instruct-v0.2": "mistral-7b",
            "codellama/CodeLlama-13b-Instruct-hf": "qwen-2.5-coder-32b",
            "llama-3.2-7b": "mistral-7b",
            "mistral-7b": "mistral-7b",
            "deepseek-r1": "deepseek-r1",
            "qwen-2.5-72b": "qwen-2.5-72b"
        }
        
        return mapping.get(model_name, "mistral-7b")  # Default to free Mistral model
    
    async def _initialize_vllm(self, model_name: str) -> bool:
        """Initialize vLLM backend"""
        if not VLLM_AVAILABLE:
            logger.error("vLLM not available. Install vLLM to use this backend.")
            return False
            
        try:
            # Check GPU requirements first
            gpu_info = VLLMServer.check_gpu_requirements()
            if not gpu_info["gpu_available"]:
                logger.error("GPU not available for vLLM")
                return False
            
            self.vllm_server = VLLMServer(model_name=model_name)
            success = await self.vllm_server.start_server()
            
            if success:
                logger.info(f"vLLM initialized with model: {model_name}")
                return True
            
        except Exception as e:
            logger.error(f"vLLM initialization failed: {e}")
        
        return False
    
    async def _initialize_ollama(self, model_name: str) -> bool:
        """Initialize Ollama backend"""
        if not OLLAMA_AVAILABLE:
            logger.error("Ollama not available. Install Ollama to use this backend.")
            return False
            
        try:
            # Check if Ollama is installed
            install_check = OllamaServer.check_installation()
            if not install_check["installed"]:
                logger.error(f"Ollama not installed: {install_check['error']}")
                return False
            
            # Map model name to Ollama format
            ollama_model = self._map_to_ollama_model(model_name)
            
            self.ollama_server = OllamaServer(model_name=ollama_model)
            success = await self.ollama_server.start_server()
            
            if success:
                logger.info(f"Ollama initialized with model: {ollama_model}")
                return True
            
        except Exception as e:
            logger.error(f"Ollama initialization failed: {e}")
        
        return False
    
    async def _initialize_openrouter_fallback(self) -> bool:
        """Fallback to OpenRouter with free models"""
        free_models = ["llama-3.2-7b", "mistral-7b", "openchat-7b"]
        
        for model in free_models:
            if await self._initialize_openrouter(model):
                self.current_backend = LLMBackend.OPENROUTER
                self.is_initialized = True
                logger.info(f"Successfully fell back to OpenRouter with model: {model}")
                return True
        
        return False
    
    async def _initialize_vllm_fallback(self) -> bool:
        """Fallback to vLLM"""
        fallback_models = self.settings.llm.fallback_models
        
        for model in fallback_models:
            if await self._initialize_vllm(model):
                self.current_backend = LLMBackend.VLLM
                self.is_initialized = True
                logger.info(f"Successfully fell back to vLLM with model: {model}")
                return True
        
        return False
    
    async def _initialize_ollama_fallback(self) -> bool:
        """Fallback to Ollama"""
        fallback_models = ["llama3.2:7b", "mistral:7b", "codellama:13b"]
        
        for model in fallback_models:
            if await self._initialize_ollama(model):
                self.current_backend = LLMBackend.OLLAMA
                self.is_initialized = True
                logger.info(f"Successfully fell back to Ollama with model: {model}")
                return True
        
        return False
    
    def _map_to_ollama_model(self, model_name: str) -> str:
        """Map standard model names to Ollama format"""
        mapping = {
            "meta-llama/Llama-3.2-7B-Instruct": "llama3.2:7b",
            "meta-llama/Llama-3.2-70B-Instruct": "llama3.2:70b",
            "mistralai/Mistral-7B-Instruct-v0.2": "mistral:7b",
            "codellama/CodeLlama-13b-Instruct-hf": "codellama:13b",
            "microsoft/Orca-2-13b": "orca2:13b"
        }
        
        return mapping.get(model_name, "llama3.2:7b")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate text using the active backend"""
        
        if not self.is_initialized:
            raise RuntimeError("LLM client not initialized. Call initialize() first.")
        
        try:
            if self.current_backend == LLMBackend.OPENROUTER:
                return await self.openrouter_client.generate(
                    prompt, max_tokens, temperature, top_p, stream, **kwargs
                )
            elif self.current_backend == LLMBackend.VLLM:
                return await self.vllm_server.generate(
                    prompt, max_tokens, temperature, top_p, stream, **kwargs
                )
            elif self.current_backend == LLMBackend.OLLAMA:
                return await self.ollama_server.generate(
                    prompt, max_tokens, temperature, top_p, stream, **kwargs
                )
            else:
                raise RuntimeError(f"Unknown backend: {self.current_backend}")
                
        except Exception as e:
            logger.error(f"Generation failed with {self.current_backend}: {e}")
            
            # Try fallback if enabled
            if self.fallback_enabled:
                return await self._generate_with_fallback(
                    prompt, max_tokens, temperature, top_p, stream, **kwargs
                )
            else:
                raise
    
    async def _generate_with_fallback(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Try generation with the other backend"""
        
        fallback_backend = (
            LLMBackend.OLLAMA if self.current_backend == LLMBackend.VLLM
            else LLMBackend.VLLM
        )
        
        logger.info(f"Attempting fallback to {fallback_backend}")
        
        try:
            if fallback_backend == LLMBackend.OPENROUTER and self.openrouter_client:
                return await self.openrouter_client.generate(
                    prompt, max_tokens, temperature, top_p, stream, **kwargs
                )
            elif fallback_backend == LLMBackend.VLLM and self.vllm_server:
                return await self.vllm_server.generate(
                    prompt, max_tokens, temperature, top_p, stream, **kwargs
                )
            elif fallback_backend == LLMBackend.OLLAMA and self.ollama_server:
                return await self.ollama_server.generate(
                    prompt, max_tokens, temperature, top_p, stream, **kwargs
                )
            else:
                # Initialize fallback backend
                if fallback_backend == LLMBackend.OPENROUTER:
                    await self._initialize_openrouter("llama-3.2-7b")
                    return await self.openrouter_client.generate(
                        prompt, max_tokens, temperature, top_p, stream, **kwargs
                    )
                elif fallback_backend == LLMBackend.VLLM:
                    await self._initialize_vllm(self.settings.llm.fallback_models[0])
                    return await self.vllm_server.generate(
                        prompt, max_tokens, temperature, top_p, stream, **kwargs
                    )
                else:
                    await self._initialize_ollama("llama3.2:7b")
                    return await self.ollama_server.generate(
                        prompt, max_tokens, temperature, top_p, stream, **kwargs
                    )
                    
        except Exception as e:
            logger.error(f"Fallback generation also failed: {e}")
            raise RuntimeError(f"Both primary and fallback backends failed: {e}")
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Chat completion using the active backend"""
        
        if not self.is_initialized:
            raise RuntimeError("LLM client not initialized. Call initialize() first.")
        
        try:
            if self.current_backend == LLMBackend.OPENROUTER:
                return await self.openrouter_client.chat(
                    messages, max_tokens, temperature, stream, **kwargs
                )
            elif self.current_backend == LLMBackend.OLLAMA:
                return await self.ollama_server.chat(
                    messages, max_tokens, temperature, stream, **kwargs
                )
            else:
                # For vLLM, convert to simple prompt format
                prompt = self._messages_to_prompt(messages)
                return await self.generate(
                    prompt, max_tokens, temperature, stream=stream, **kwargs
                )
                
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt"""
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)
    
    async def health_check(self) -> Dict:
        """Check health of all backends"""
        health_status = {
            "current_backend": self.current_backend.value if self.current_backend else None,
            "is_initialized": self.is_initialized,
            "backends": {}
        }
        
        # Check OpenRouter
        if self.openrouter_client:
            try:
                openrouter_healthy = await self.openrouter_client.health_check()
                health_status["backends"]["openrouter"] = {
                    "healthy": openrouter_healthy,
                    "model_info": await self.openrouter_client.get_model_info()
                }
            except Exception as e:
                health_status["backends"]["openrouter"] = {
                    "healthy": False,
                    "error": str(e)
                }
        
        # Check vLLM
        if self.vllm_server:
            try:
                vllm_healthy = await self.vllm_server.health_check()
                health_status["backends"]["vllm"] = {
                    "healthy": vllm_healthy,
                    "model_info": await self.vllm_server.get_model_info()
                }
            except Exception as e:
                health_status["backends"]["vllm"] = {
                    "healthy": False,
                    "error": str(e)
                }
        
        # Check Ollama
        if self.ollama_server:
            try:
                ollama_healthy = await self.ollama_server.health_check()
                health_status["backends"]["ollama"] = {
                    "healthy": ollama_healthy,
                    "model_info": await self.ollama_server.get_model_info()
                }
            except Exception as e:
                health_status["backends"]["ollama"] = {
                    "healthy": False,
                    "error": str(e)
                }
        
        return health_status
    
    async def switch_backend(self, backend: LLMBackend, model_name: str = None) -> bool:
        """Switch to a different backend"""
        try:
            if backend == LLMBackend.OPENROUTER:
                success = await self._initialize_openrouter(
                    model_name or "mistral-7b"
                )
            elif backend == LLMBackend.VLLM:
                success = await self._initialize_vllm(
                    model_name or self.settings.llm.model_name
                )
            else:
                success = await self._initialize_ollama(
                    model_name or "llama3.2:7b"
                )
            
            if success:
                self.current_backend = backend
                logger.info(f"Successfully switched to {backend.value}")
                return True
            
        except Exception as e:
            logger.error(f"Failed to switch to {backend.value}: {e}")
        
        return False
    
    async def shutdown(self):
        """Shutdown all backends"""
        try:
            if self.vllm_server:
                await self.vllm_server.stop_server()
            
            if self.ollama_server:
                await self.ollama_server.stop_server()
            
            self.is_initialized = False
            self.current_backend = None
            
            logger.info("LLM client shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def get_available_models(self) -> Dict[str, Dict]:
        """Get available models for all backends"""
        models = {}
        
        # OpenRouter models
        models["openrouter"] = OpenRouterClient.get_available_models()
        
        # vLLM models (if available)
        if VLLM_AVAILABLE and VLLMServer:
            models["vllm"] = VLLMServer.get_available_models()
        else:
            models["vllm"] = {}
            
        # Ollama models (if available)
        if OLLAMA_AVAILABLE and OllamaServer:
            models["ollama"] = OllamaServer.get_available_models()
        else:
            models["ollama"] = {}
            
        return models