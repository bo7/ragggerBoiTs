"""
vLLM Server Implementation
=========================

High-performance LLM serving using vLLM with support for multiple open-source models.
"""

import asyncio
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, AsyncGenerator
import httpx
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import torch

from ..config import get_settings

logger = logging.getLogger(__name__)


class VLLMServer:
    """vLLM server for local LLM inference"""
    
    SUPPORTED_MODELS = {
        "llama4": "meta-llama/Llama-3.2-7B-Instruct",
        "llama4-large": "meta-llama/Llama-3.2-70B-Instruct", 
        "falcon": "tiiuae/falcon-180B-chat",
        "falcon-7b": "tiiuae/falcon-7b-instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "mistral-large": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "orca": "microsoft/Orca-2-13b",
        "codellama": "codellama/CodeLlama-13b-Instruct-hf"
    }
    
    def __init__(self, model_name: str = None, **kwargs):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.llm.model_name
        self.host = self.settings.llm.host
        self.port = self.settings.llm.port
        
        # vLLM configuration
        self.gpu_memory_utilization = kwargs.get(
            'gpu_memory_utilization', 
            self.settings.llm.gpu_memory_utilization
        )
        self.max_model_len = kwargs.get(
            'max_model_len',
            self.settings.llm.max_model_len
        )
        self.tensor_parallel_size = kwargs.get('tensor_parallel_size', 1)
        
        self.engine: Optional[AsyncLLMEngine] = None
        self.server_process: Optional[subprocess.Popen] = None
        self.is_running = False
        
    async def start_server(self, background: bool = True) -> bool:
        """Start the vLLM server"""
        try:
            if background:
                await self._start_background_server()
            else:
                await self._start_inline_engine()
            
            self.is_running = True
            logger.info(f"vLLM server started with model: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start vLLM server: {e}")
            return False
    
    async def _start_background_server(self):
        """Start vLLM as a background API server"""
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_name,
            "--host", self.host,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-model-len", str(self.max_model_len),
        ]
        
        if self.tensor_parallel_size > 1:
            cmd.extend(["--tensor-parallel-size", str(self.tensor_parallel_size)])
        
        logger.info(f"Starting vLLM server with command: {' '.join(cmd)}")
        
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to be ready
        await self._wait_for_server()
    
    async def _start_inline_engine(self):
        """Start vLLM engine inline (for direct Python API)"""
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            tensor_parallel_size=self.tensor_parallel_size,
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("vLLM async engine started inline")
    
    async def _wait_for_server(self, timeout: int = 300):
        """Wait for the vLLM server to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"http://{self.host}:{self.port}/health")
                    if response.status_code == 200:
                        logger.info("vLLM server is ready")
                        return
            except:
                pass
            
            await asyncio.sleep(2)
        
        raise TimeoutError(f"vLLM server failed to start within {timeout} seconds")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate text using the loaded model"""
        
        if self.engine:
            # Direct engine generation
            return await self._generate_with_engine(
                prompt, max_tokens, temperature, top_p, stream, **kwargs
            )
        else:
            # API server generation
            return await self._generate_with_api(
                prompt, max_tokens, temperature, top_p, stream, **kwargs
            )
    
    async def _generate_with_engine(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate using direct engine access"""
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        if stream:
            return self._stream_generate_engine(prompt, sampling_params)
        else:
            results = await self.engine.generate(prompt, sampling_params)
            return results[0].outputs[0].text
    
    async def _stream_generate_engine(
        self, 
        prompt: str, 
        sampling_params: SamplingParams
    ) -> AsyncGenerator[str, None]:
        """Stream generation using engine"""
        
        results_generator = self.engine.generate(
            prompt, sampling_params, request_id=f"req_{time.time()}"
        )
        
        async for request_output in results_generator:
            yield request_output.outputs[0].text
    
    async def _generate_with_api(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate using API server"""
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            **kwargs
        }
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            if stream:
                return self._stream_generate_api(client, payload)
            else:
                response = await client.post(
                    f"http://{self.host}:{self.port}/v1/chat/completions",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
    
    async def _stream_generate_api(
        self, 
        client: httpx.AsyncClient, 
        payload: Dict
    ) -> AsyncGenerator[str, None]:
        """Stream generation using API"""
        
        async with client.stream(
            "POST",
            f"http://{self.host}:{self.port}/v1/chat/completions",
            json=payload
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    if line == "data: [DONE]":
                        break
                    
                    try:
                        import json
                        data = json.loads(line[6:])
                        if data["choices"][0]["delta"].get("content"):
                            yield data["choices"][0]["delta"]["content"]
                    except (json.JSONDecodeError, KeyError):
                        continue
    
    async def health_check(self) -> bool:
        """Check if the server is healthy"""
        try:
            if self.engine:
                return True
            
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://{self.host}:{self.port}/health")
                return response.status_code == 200
        except:
            return False
    
    async def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        try:
            if self.engine:
                return {
                    "model_name": self.model_name,
                    "max_model_len": self.max_model_len,
                    "gpu_memory_utilization": self.gpu_memory_utilization,
                    "status": "running"
                }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://{self.host}:{self.port}/v1/models")
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}
    
    async def stop_server(self):
        """Stop the vLLM server"""
        try:
            if self.server_process:
                self.server_process.terminate()
                self.server_process.wait(timeout=30)
                logger.info("vLLM server stopped")
            
            if self.engine:
                # The engine doesn't have a direct stop method
                self.engine = None
                logger.info("vLLM engine stopped")
            
            self.is_running = False
            
        except Exception as e:
            logger.error(f"Error stopping vLLM server: {e}")
    
    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """Get list of supported models"""
        return cls.SUPPORTED_MODELS.copy()
    
    @staticmethod
    def check_gpu_requirements() -> Dict:
        """Check GPU requirements and availability"""
        if not torch.cuda.is_available():
            return {
                "gpu_available": False,
                "error": "CUDA not available"
            }
        
        gpu_count = torch.cuda.device_count()
        gpu_memory = []
        
        for i in range(gpu_count):
            memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            gpu_memory.append(f"GPU {i}: {memory:.1f}GB")
        
        return {
            "gpu_available": True,
            "gpu_count": gpu_count,
            "gpu_memory": gpu_memory,
            "recommendations": {
                "7B models": "Minimum 16GB VRAM",
                "13B models": "Minimum 24GB VRAM", 
                "70B models": "Minimum 80GB VRAM (multiple GPUs)",
                "180B models": "Minimum 160GB VRAM (multiple GPUs)"
            }
        }