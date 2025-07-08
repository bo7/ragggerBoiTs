"""
Jina v4 Embedding Client
========================

Cloud API integration for Jina v4 embedding service.
Provides multimodal embeddings for text, images, and documents.
"""

import asyncio
import base64
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import httpx
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings

logger = logging.getLogger(__name__)


class JinaEmbeddingClient:
    """Client for Jina v4 cloud embedding service"""
    
    def __init__(self, 
                 api_key: str = None,
                 model_name: str = "jina-embeddings-v4",
                 timeout: float = 30.0):
        self.api_key = api_key or self._get_api_key()
        self.model_name = model_name
        self.base_url = "https://api.jina.ai/v1"
        self.timeout = timeout
        self.session: Optional[httpx.AsyncClient] = None
        
        # Model configurations (updated for Jina v4)
        self.supported_models = {
            "jina-embeddings-v4": {
                "dimension": 2048,
                "max_seq_length": 8192,
                "supports_multimodal": True,
                "supports_late_chunking": True,
                "supports_tasks": ["retrieval.query", "retrieval.passage", "code.query", "code.passage", "text-matching"]
            },
            "jina-embeddings-v3": {
                "dimension": 1024,
                "max_seq_length": 8192,
                "supports_multimodal": True,
                "supports_late_chunking": True,
                "supports_tasks": ["retrieval.query", "retrieval.passage", "text-matching"]
            },
            "jina-clip-v1": {
                "dimension": 768,
                "max_seq_length": 77,
                "supports_multimodal": True,
                "supports_late_chunking": False,
                "supports_tasks": ["retrieval.query", "retrieval.passage"]
            },
            "jina-colbert-v2": {
                "dimension": 128,
                "max_seq_length": 8192,
                "supports_multimodal": False,
                "supports_late_chunking": True,
                "supports_tasks": ["retrieval.query", "retrieval.passage"]
            }
        }
        
        self.is_initialized = False
        
    def _get_api_key(self) -> str:
        """Get Jina API key from environment or settings"""
        # Try environment variables
        api_key = os.getenv('JINA_API_KEY')
        if api_key:
            return api_key
        
        # Try settings
        settings = get_settings()
        if hasattr(settings, 'jina_api_key'):
            return settings.jina_api_key
        
        raise ValueError(
            "Jina API key not found. Set JINA_API_KEY environment variable "
            "or add jina_api_key to settings. Get your API key from: https://jina.ai/embeddings/"
        )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers for Jina API"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    async def initialize(self) -> bool:
        """Initialize the Jina embedding client"""
        try:
            self.session = httpx.AsyncClient(timeout=self.timeout)
            
            # Check if Jina API is accessible
            if await self.health_check():
                self.is_initialized = True
                logger.info(f"Jina embedding client initialized with model: {self.model_name}")
                return True
            else:
                logger.error("Jina embedding API is not accessible")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize Jina embedding client: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check if Jina embedding API is accessible"""
        try:
            if not self.session:
                self.session = httpx.AsyncClient(timeout=self.timeout)
                
            # Test with a simple embedding request
            response = await self.session.post(
                f"{self.base_url}/embeddings",
                headers=self._get_headers(),
                json={
                    "model": self.model_name,
                    "input": ["test"],
                    "task": "text-matching"
                }
            )
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def embed_text(self, 
                        texts: Union[str, List[str]], 
                        task: str = "retrieval.passage",
                        late_chunking: bool = False,
                        **kwargs) -> List[List[float]]:
        """
        Embed text using Jina v4
        
        Args:
            texts: Single text or list of texts to embed
            task: Task type (retrieval.passage, retrieval.query, classification, etc.)
            late_chunking: Whether to use late chunking for long texts
            **kwargs: Additional parameters
            
        Returns:
            List of embeddings
        """
        if not self.is_initialized:
            raise RuntimeError("Jina client not initialized. Call initialize() first.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        payload = {
            "model": self.model_name,
            "input": texts,
            "task": task,
            "late_chunking": late_chunking,
            **kwargs
        }
        
        try:
            response = await self.session.post(
                f"{self.base_url}/embeddings",
                headers=self._get_headers(),
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            
            if "error" in data:
                raise RuntimeError(f"Jina API error: {data['error']}")
            
            embeddings = []
            for item in data.get("data", []):
                embeddings.append(item.get("embedding", []))
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def embed_image(self, 
                         images: Union[str, bytes, List[Union[str, bytes]]],
                         task: str = "retrieval.passage",
                         **kwargs) -> List[List[float]]:
        """
        Embed images using Jina v4
        
        Args:
            images: Single image path/bytes or list of images
            task: Task type
            **kwargs: Additional parameters
            
        Returns:
            List of embeddings
        """
        if not self.is_initialized:
            raise RuntimeError("Jina client not initialized. Call initialize() first.")
        
        if not self.supported_models[self.model_name]["supports_multimodal"]:
            raise RuntimeError(f"Model {self.model_name} does not support multimodal embeddings")
        
        if not isinstance(images, list):
            images = [images]
        
        # Process images to base64
        processed_images = []
        for image in images:
            if isinstance(image, str):
                # Assume it's a file path
                with open(image, "rb") as f:
                    image_bytes = f.read()
            else:
                image_bytes = image
            
            # Convert to base64
            image_b64 = base64.b64encode(image_bytes).decode()
            processed_images.append(f"data:image/jpeg;base64,{image_b64}")
        
        payload = {
            "model": self.model_name,
            "input": processed_images,
            "task": task,
            **kwargs
        }
        
        try:
            response = await self.session.post(
                f"{self.base_url}/embeddings",
                headers=self._get_headers(),
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            
            if "error" in data:
                raise RuntimeError(f"Jina API error: {data['error']}")
            
            embeddings = []
            for item in data.get("data", []):
                embeddings.append(item.get("embedding", []))
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Image embedding failed: {e}")
            raise
    
    async def embed_documents(self, 
                             documents: List[Dict[str, Any]],
                             task: str = "retrieval.passage",
                             **kwargs) -> List[Dict[str, Any]]:
        """
        Embed documents with mixed content (text + images)
        
        Args:
            documents: List of documents with 'text' and/or 'image' fields
            task: Task type
            **kwargs: Additional parameters
            
        Returns:
            List of documents with embeddings
        """
        results = []
        
        for doc in documents:
            result = {"document": doc, "embeddings": {}}
            
            # Embed text if present
            if "text" in doc and doc["text"]:
                text_embeddings = await self.embed_text(
                    doc["text"], task=task, **kwargs
                )
                result["embeddings"]["text"] = text_embeddings[0]
            
            # Embed image if present
            if "image" in doc and doc["image"]:
                image_embeddings = await self.embed_image(
                    doc["image"], task=task, **kwargs
                )
                result["embeddings"]["image"] = image_embeddings[0]
            
            results.append(result)
        
        return results
    
    async def compute_similarity(self, 
                               embedding1: List[float], 
                               embedding2: List[float],
                               metric: str = "cosine") -> float:
        """
        Compute similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric (cosine, dot, euclidean)
            
        Returns:
            Similarity score
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            if metric == "cosine":
                # Cosine similarity
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                return dot_product / (norm1 * norm2)
            
            elif metric == "dot":
                # Dot product
                return np.dot(vec1, vec2)
            
            elif metric == "euclidean":
                # Euclidean distance (converted to similarity)
                distance = np.linalg.norm(vec1 - vec2)
                return 1 / (1 + distance)
            
            else:
                raise ValueError(f"Unsupported metric: {metric}")
                
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            raise
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        try:
            if not self.session:
                return {"error": "Client not initialized"}
            
            response = await self.session.get(f"{self.base_url}/models")
            
            if response.status_code == 200:
                data = response.json()
                
                # Find current model in the list
                for model in data.get("data", []):
                    if model.get("id") == self.model_name:
                        return {
                            "model_name": self.model_name,
                            "dimension": self.supported_models.get(self.model_name, {}).get("dimension", "unknown"),
                            "max_seq_length": self.supported_models.get(self.model_name, {}).get("max_seq_length", "unknown"),
                            "supports_multimodal": self.supported_models.get(self.model_name, {}).get("supports_multimodal", False),
                            "supports_late_chunking": self.supported_models.get(self.model_name, {}).get("supports_late_chunking", False),
                            "status": "available"
                        }
            
            return {
                "model_name": self.model_name,
                "status": "not_found",
                "error": f"Model {self.model_name} not found"
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch to a different embedding model"""
        if model_name not in self.supported_models:
            logger.error(f"Model {model_name} not supported. Available models: {list(self.supported_models.keys())}")
            return False
        
        self.model_name = model_name
        logger.info(f"Switched to model: {model_name}")
        return True
    
    def get_supported_models(self) -> Dict[str, Dict]:
        """Get list of supported models"""
        return self.supported_models.copy()
    
    async def batch_embed(self, 
                         inputs: List[Union[str, bytes]],
                         batch_size: int = 32,
                         task: str = "retrieval.passage",
                         **kwargs) -> List[List[float]]:
        """
        Embed large batches of inputs efficiently
        
        Args:
            inputs: List of text strings or image bytes
            batch_size: Number of items to process in each batch
            task: Task type
            **kwargs: Additional parameters
            
        Returns:
            List of embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            
            # Determine if batch contains text or images
            if all(isinstance(item, str) for item in batch):
                # Text batch
                embeddings = await self.embed_text(batch, task=task, **kwargs)
            else:
                # Image batch
                embeddings = await self.embed_image(batch, task=task, **kwargs)
            
            all_embeddings.extend(embeddings)
            
            # Add small delay to avoid overwhelming the service
            await asyncio.sleep(0.1)
        
        return all_embeddings
    
    async def shutdown(self):
        """Shutdown the client"""
        if self.session:
            await self.session.aclose()
            self.session = None
        
        self.is_initialized = False
        logger.info("Jina embedding client shutdown complete")


    async def test_embedding(self, 
                           test_text: str = "Hello, world!",
                           task: str = "text-matching") -> Dict[str, Any]:
        """Test the Jina embedding service with a simple text"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            embeddings = await self.embed_text(
                test_text,
                task=task
            )
            
            end_time = asyncio.get_event_loop().time()
            response_time = end_time - start_time
            
            return {
                "success": True,
                "model": self.model_name,
                "text": test_text,
                "embedding_dimension": len(embeddings[0]) if embeddings else 0,
                "response_time": response_time,
                "task": task
            }
            
        except Exception as e:
            return {
                "success": False,
                "model": self.model_name,
                "error": str(e),
                "text": test_text,
                "task": task
            }
    
    def get_pricing_info(self) -> Dict[str, Any]:
        """Get pricing information for Jina API"""
        return {
            "free_tokens": "10 million tokens for new users",
            "pricing_model": "Pay-as-you-go after free tier",
            "billing_date": "New pricing model introduced May 6th, 2025",
            "get_api_key": "https://jina.ai/embeddings/",
            "note": "Pricing varies by model and usage volume"
        }