"""
Open-Source RAG System with Agent Framework
===========================================

A comprehensive retrieval-augmented generation system built entirely with open-source components:
- vLLM for local LLM serving (Llama 4, Falcon 180B, Mistral)
- Jina v4 for multimodal embeddings
- Milvus for vector storage
- Neo4j for graph knowledge
- AutoGen v0.4 for agent orchestration
- Custom MCP servers for tool integration

Designed for customer deployment with complete data sovereignty.
"""

__version__ = "1.0.0"
__author__ = "RAG Systems Team"

# Core components
from .llm_serving import LLMClient, LLMBackend, OpenRouterClient
from .config import get_settings, settings