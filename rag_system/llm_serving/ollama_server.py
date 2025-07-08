"""
Ollama Server Implementation
===========================

Simple LLM serving using Ollama for development and lightweight deployments.
"""

import asyncio
import json
import logging
import subprocess
import time
from typing import Dict, List, Optional, Union, AsyncGenerator
import httpx

from ..config import get_settings

logger = logging.getLogger(__name__)


class OllamaServer:
    """Ollama server for simple local LLM inference"""
    
    SUPPORTED_MODELS = {
        "llama3.2": "llama3.2:7b",
        "llama3.2-large": "llama3.2:70b",
        "mistral": "mistral:7b",
        "mistral-large": "mistral:latest",
        "codellama": "codellama:13b",
        "falcon": "falcon:7b",
        "orca2": "orca2:13b"
    }
    
    def __init__(self, model_name: str = None):
        self.settings = get_settings()
        self.model_name = model_name or "llama3.2:7b"
        self.host = self.settings.llm.host
        self.port = 11434  # Ollama default port
        self.is_running = False
        self.server_process: Optional[subprocess.Popen] = None
        
    async def start_server(self) -> bool:
        """Start the Ollama server"""
        try:
            # Check if Ollama is already running
            if await self._is_ollama_running():
                logger.info("Ollama server is already running")
                self.is_running = True
                return True
            
            # Start Ollama server
            await self._start_ollama_service()
            
            # Pull the model if not available
            await self._ensure_model_available()
            
            self.is_running = True
            logger.info(f"Ollama server started with model: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Ollama server: {e}")
            return False
    
    async def _is_ollama_running(self) -> bool:
        """Check if Ollama server is running"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://{self.host}:{self.port}/api/tags")
                return response.status_code == 200
        except:
            return False
    
    async def _start_ollama_service(self):
        """Start the Ollama service"""
        try:
            # Try to start Ollama
            result = subprocess.run(
                ["ollama", "serve"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Wait for server to be ready
            await self._wait_for_server()
            
        except subprocess.TimeoutExpired:
            # This is expected as serve runs in background
            await self._wait_for_server()
        except FileNotFoundError:
            raise RuntimeError("Ollama is not installed. Please install Ollama first.")
    
    async def _wait_for_server(self, timeout: int = 60):
        """Wait for Ollama server to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if await self._is_ollama_running():
                logger.info("Ollama server is ready")
                return
            await asyncio.sleep(2)
        
        raise TimeoutError(f"Ollama server failed to start within {timeout} seconds")
    
    async def _ensure_model_available(self):
        """Ensure the specified model is available"""
        try:
            # Check if model is already available
            available_models = await self.list_models()
            
            if self.model_name not in [model["name"] for model in available_models]:
                logger.info(f"Pulling model: {self.model_name}")
                await self._pull_model(self.model_name)
            else:
                logger.info(f"Model {self.model_name} is already available")
                
        except Exception as e:
            logger.error(f"Failed to ensure model availability: {e}")
            raise
    
    async def _pull_model(self, model_name: str):
        """Pull a model from Ollama repository"""
        try:
            process = await asyncio.create_subprocess_exec(
                "ollama", "pull", model_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"Successfully pulled model: {model_name}")
            else:
                raise RuntimeError(f"Failed to pull model {model_name}: {stderr.decode()}")
                
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            raise
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate text using Ollama"""
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                **kwargs
            }
        }
        
        if stream:
            return self._stream_generate(payload)
        else:
            return await self._generate_sync(payload)
    
    async def _generate_sync(self, payload: Dict) -> str:
        """Synchronous generation"""
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"http://{self.host}:{self.port}/api/generate",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
    
    async def _stream_generate(self, payload: Dict) -> AsyncGenerator[str, None]:
        """Stream generation"""
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"http://{self.host}:{self.port}/api/generate",
                json=payload
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                            if data.get("done", False):
                                break
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
        """Chat completion using Ollama"""
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                **kwargs
            }
        }
        
        if stream:
            return self._stream_chat(payload)
        else:
            return await self._chat_sync(payload)
    
    async def _chat_sync(self, payload: Dict) -> str:
        """Synchronous chat"""
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"http://{self.host}:{self.port}/api/chat",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
    
    async def _stream_chat(self, payload: Dict) -> AsyncGenerator[str, None]:
        """Stream chat"""
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"http://{self.host}:{self.port}/api/chat",
                json=payload
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                yield data["message"]["content"]
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
    
    async def list_models(self) -> List[Dict]:
        """List available models"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://{self.host}:{self.port}/api/tags")
                response.raise_for_status()
                data = response.json()
                return data.get("models", [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check if Ollama server is healthy"""
        return await self._is_ollama_running()
    
    async def get_model_info(self) -> Dict:
        """Get information about the current model"""
        try:
            models = await self.list_models()
            for model in models:
                if model["name"] == self.model_name:
                    return {
                        "model_name": self.model_name,
                        "size": model.get("size", "unknown"),
                        "modified_at": model.get("modified_at", "unknown"),
                        "status": "running"
                    }
            
            return {
                "model_name": self.model_name,
                "status": "not_found",
                "available_models": [m["name"] for m in models]
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}
    
    async def stop_server(self):
        """Stop Ollama server (note: this stops the system service)"""
        try:
            # Ollama runs as a system service, so we don't typically stop it
            # But we can mark our connection as stopped
            self.is_running = False
            logger.info("Ollama connection stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Ollama: {e}")
    
    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """Get list of supported models"""
        return cls.SUPPORTED_MODELS.copy()
    
    @staticmethod
    def check_installation() -> Dict:
        """Check if Ollama is installed and accessible"""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return {
                    "installed": True,
                    "version": result.stdout.strip(),
                    "status": "ready"
                }
            else:
                return {
                    "installed": False,
                    "error": "Ollama command failed",
                    "status": "error"
                }
                
        except FileNotFoundError:
            return {
                "installed": False,
                "error": "Ollama not found in PATH",
                "status": "not_installed",
                "install_instructions": "Visit https://ollama.ai/download"
            }
        except subprocess.TimeoutExpired:
            return {
                "installed": False,
                "error": "Ollama command timed out",
                "status": "timeout"
            }