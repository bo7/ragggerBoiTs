"""
OpenRouter Client Implementation
===============================

LLM serving using OpenRouter API with open-source models.
This provides immediate access to various open-source models without local setup.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Union, AsyncGenerator
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """OpenRouter client for accessing open-source models via API"""
    
    # Focus on open-source models available on OpenRouter (updated with correct IDs)
    OPEN_SOURCE_MODELS = {
        # Meta Llama models (FREE)
        "llama-4-maverick": "meta-llama/llama-4-maverick:free",
        "llama-4-scout": "meta-llama/llama-4-scout:free", 
        "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct:free",
        "llama-3.2-11b-vision": "meta-llama/llama-3.2-11b-vision-instruct:free",
        
        # Mistral models (FREE)
        "mistral-7b": "mistralai/mistral-7b-instruct:free",
        "mistral-nemo": "mistralai/mistral-nemo:free",
        "mistral-small-24b": "mistralai/mistral-small-24b-instruct-2501:free",
        "mistral-small-31": "mistralai/mistral-small-3.1-24b-instruct:free",
        "devstral-small": "mistralai/devstral-small:free",
        
        # DeepSeek models (FREE)
        "deepseek-r1": "deepseek/deepseek-r1:free",
        "deepseek-chat": "deepseek/deepseek-chat:free",
        "deepseek-v3": "deepseek/deepseek-v3-base:free",
        "deepseek-r1-distill-14b": "deepseek/deepseek-r1-distill-qwen-14b:free",
        "deepseek-r1-distill-70b": "deepseek/deepseek-r1-distill-llama-70b:free",
        
        # Qwen models (FREE)
        "qwen-2.5-72b": "qwen/qwen-2.5-72b-instruct:free",
        "qwen-2.5-coder-32b": "qwen/qwen-2.5-coder-32b-instruct:free",
        "qwen-2.5-vl-32b": "qwen/qwen2.5-vl-32b-instruct:free",
        "qwen-2.5-vl-72b": "qwen/qwen2.5-vl-72b-instruct:free",
        "qwq-32b": "qwen/qwq-32b:free",
        "qwen3-8b": "qwen/qwen3-8b:free",
        "qwen3-14b": "qwen/qwen3-14b:free",
        "qwen3-32b": "qwen/qwen3-32b:free",
        
        # Google models (FREE)
        "gemma-2-9b": "google/gemma-2-9b-it:free",
        "gemma-3-4b": "google/gemma-3-4b-it:free",
        "gemma-3-12b": "google/gemma-3-12b-it:free",
        "gemma-3-27b": "google/gemma-3-27b-it:free",
        "gemini-2.0-flash": "google/gemini-2.0-flash-exp:free",
        
        # Other free models
        "nvidia-nemotron-49b": "nvidia/llama-3.3-nemotron-super-49b-v1:free",
        "nvidia-nemotron-253b": "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
        "reka-flash-3": "rekaai/reka-flash-3:free",
        "dolphin-mistral": "cognitivecomputations/dolphin3.0-mistral-24b:free"
    }
    
    def __init__(self, api_key: str = None, model_name: str = None):
        self.api_key = api_key or self._get_api_key()
        self.model_name = model_name or "mistralai/mistral-7b-instruct:free"
        self.base_url = "https://openrouter.ai/api/v1"
        
        # Rate limiting
        self.rate_limit_remaining = 100
        self.rate_limit_reset = None
        
        # Client configuration
        self.timeout = 120.0
        self.max_retries = 3
        
    def _get_api_key(self) -> str:
        """Get OpenRouter API key from environment or settings"""
        import os
        
        # Try environment variables
        api_key = os.getenv('OPENROUTER_API_KEY')
        if api_key:
            return api_key
        
        # Try settings
        settings = get_settings()
        if hasattr(settings, 'openrouter_api_key'):
            return settings.openrouter_api_key
        
        raise ValueError(
            "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable "
            "or add openrouter_api_key to settings."
        )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/bo7/ragggerBoiTs",
            "X-Title": "RAG System"
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate text using OpenRouter"""
        
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            **kwargs
        }
        
        if stream:
            return self._stream_generate(payload)
        else:
            return await self._generate_sync(payload)
    
    async def _generate_sync(self, payload: Dict) -> str:
        """Synchronous generation"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=payload
            )
            
            self._update_rate_limits(response.headers)
            
            # Debug logging
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"OpenRouter API error {response.status_code}: {error_text}")
                raise RuntimeError(f"OpenRouter API error {response.status_code}: {error_text}")
            
            response.raise_for_status()
            
            data = response.json()
            
            if "error" in data:
                raise RuntimeError(f"OpenRouter API error: {data['error']}")
            
            return data["choices"][0]["message"]["content"]
    
    async def _stream_generate(self, payload: Dict) -> AsyncGenerator[str, None]:
        """Stream generation"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=payload
            ) as response:
                
                self._update_rate_limits(response.headers)
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        if line == "data: [DONE]":
                            break
                        
                        try:
                            data = json.loads(line[6:])
                            if "error" in data:
                                raise RuntimeError(f"OpenRouter API error: {data['error']}")
                            
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                                
                        except json.JSONDecodeError:
                            continue
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Chat completion using OpenRouter"""
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            **kwargs
        }
        
        if stream:
            return self._stream_generate(payload)
        else:
            return await self._generate_sync(payload)
    
    def _update_rate_limits(self, headers: Dict):
        """Update rate limit information from response headers"""
        self.rate_limit_remaining = int(headers.get("x-ratelimit-remaining", 100))
        self.rate_limit_reset = headers.get("x-ratelimit-reset-date")
    
    async def get_models(self) -> List[Dict]:
        """Get available models from OpenRouter"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=self._get_headers()
                )
                response.raise_for_status()
                
                data = response.json()
                return data.get("data", [])
                
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return []
    
    async def get_model_info(self) -> Dict:
        """Get information about the current model"""
        try:
            models = await self.get_models()
            
            for model in models:
                if model["id"] == self.model_name:
                    return {
                        "model_name": self.model_name,
                        "description": model.get("description", ""),
                        "context_length": model.get("context_length", "unknown"),
                        "pricing": model.get("pricing", {}),
                        "status": "available"
                    }
            
            return {
                "model_name": self.model_name,
                "status": "not_found",
                "error": f"Model {self.model_name} not found in available models"
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> bool:
        """Check if OpenRouter API is accessible"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=self._get_headers()
                )
                return response.status_code == 200
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def get_usage_stats(self) -> Dict:
        """Get usage statistics (if supported by OpenRouter)"""
        try:
            return {
                "rate_limit_remaining": self.rate_limit_remaining,
                "rate_limit_reset": self.rate_limit_reset,
                "current_model": self.model_name
            }
        except Exception as e:
            logger.error(f"Failed to get usage stats: {e}")
            return {"error": str(e)}
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model"""
        # Handle both direct model names and our aliases
        if model_name in self.OPEN_SOURCE_MODELS:
            self.model_name = self.OPEN_SOURCE_MODELS[model_name]
            logger.info(f"Switched to model: {self.model_name} (from alias: {model_name})")
            return True
        elif model_name in self.OPEN_SOURCE_MODELS.values():
            self.model_name = model_name
            logger.info(f"Switched to model: {self.model_name}")
            return True
        else:
            logger.error(f"Model {model_name} not supported. Available aliases: {list(self.OPEN_SOURCE_MODELS.keys())}")
            return False
    
    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """Get list of supported open-source models"""
        return cls.OPEN_SOURCE_MODELS.copy()
    
    @classmethod 
    def get_free_models(cls) -> Dict[str, str]:
        """Get list of free open-source models"""
        return {
            alias: model for alias, model in cls.OPEN_SOURCE_MODELS.items()
            if ":free" in model
        }
    
    def get_cost_estimate(self, input_tokens: int, output_tokens: int = None) -> Dict:
        """Estimate cost for a request (basic estimation)"""
        # This is a rough estimate - actual pricing varies by model
        free_models = self.get_free_models()
        
        if any(self.model_name == model for model in free_models.values()):
            return {
                "estimated_cost": 0.0,
                "currency": "USD",
                "model": self.model_name,
                "note": "This is a free model"
            }
        else:
            # Rough estimate for paid models (you'd want to get actual pricing)
            estimated_cost = (input_tokens * 0.000001) + ((output_tokens or 512) * 0.000002)
            return {
                "estimated_cost": estimated_cost,
                "currency": "USD", 
                "model": self.model_name,
                "note": "This is a rough estimate. Check OpenRouter for actual pricing."
            }
    
    async def test_model(self, test_prompt: str = "Hello! Please respond with 'Hello, World!'") -> Dict:
        """Test the current model with a simple prompt"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            response = await self.generate(
                test_prompt,
                max_tokens=50,
                temperature=0.1
            )
            
            end_time = asyncio.get_event_loop().time()
            response_time = end_time - start_time
            
            return {
                "success": True,
                "model": self.model_name,
                "response": response,
                "response_time": response_time,
                "test_prompt": test_prompt
            }
            
        except Exception as e:
            return {
                "success": False,
                "model": self.model_name,
                "error": str(e),
                "test_prompt": test_prompt
            }