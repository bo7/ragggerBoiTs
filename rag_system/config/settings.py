"""
Configuration settings for the open-source RAG system
"""
import os
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class LLMSettings(BaseSettings):
    """LLM serving configuration"""
    backend: str = Field("openrouter", description="LLM backend: openrouter, vllm, or ollama")
    model_name: str = Field("llama-3.2-7b", description="Primary model")
    fallback_models: List[str] = Field(
        default=["mistral-7b", "openchat-7b"],
        description="Fallback models"
    )
    
    # OpenRouter settings
    openrouter_api_key: Optional[str] = Field(None, description="OpenRouter API key")
    
    # Local server settings (for vLLM/Ollama)
    host: str = Field("localhost", description="LLM server host")
    port: int = Field(8000, description="LLM server port")
    gpu_memory_utilization: float = Field(0.8, description="GPU memory utilization")
    max_model_len: int = Field(32768, description="Maximum sequence length")
    
    class Config:
        env_prefix = "LLM_"


class EmbeddingSettings(BaseSettings):
    """Jina v4 embedding configuration"""
    model_name: str = Field("jinaai/jina-embeddings-v4", description="Jina v4 model")
    host: str = Field("localhost", description="Embedding server host")
    port: int = Field(8001, description="Embedding server port")
    batch_size: int = Field(32, description="Embedding batch size")
    max_tokens: int = Field(32000, description="Maximum tokens per document")
    dimensions: int = Field(2048, description="Embedding dimensions")
    late_chunking: bool = Field(True, description="Enable late chunking")
    multimodal: bool = Field(True, description="Enable multimodal processing")
    
    class Config:
        env_prefix = "EMBEDDING_"


class VectorDBSettings(BaseSettings):
    """Milvus vector database configuration"""
    host: str = Field("localhost", description="Milvus host")
    port: int = Field(19530, description="Milvus port")
    collection_name: str = Field("rag_documents", description="Default collection")
    index_type: str = Field("IVF_FLAT", description="Vector index type")
    metric_type: str = Field("COSINE", description="Distance metric")
    nlist: int = Field(1024, description="Number of cluster units")
    
    class Config:
        env_prefix = "VECTOR_DB_"


class GraphDBSettings(BaseSettings):
    """Neo4j graph database configuration"""
    uri: str = Field("neo4j://localhost:7687", description="Neo4j URI")
    username: str = Field("neo4j", description="Neo4j username")
    password: str = Field("password", description="Neo4j password")
    database: str = Field("neo4j", description="Neo4j database name")
    
    class Config:
        env_prefix = "GRAPH_DB_"


class AgentSettings(BaseSettings):
    """AutoGen agent configuration"""
    max_agents: int = Field(10, description="Maximum concurrent agents")
    agent_timeout: int = Field(300, description="Agent timeout in seconds")
    enable_code_execution: bool = Field(False, description="Allow code execution")
    enable_human_feedback: bool = Field(True, description="Enable human in the loop")
    
    class Config:
        env_prefix = "AGENT_"


class MCPSettings(BaseSettings):
    """MCP server configuration"""
    servers_dir: str = Field("mcp_servers", description="MCP servers directory")
    max_connections: int = Field(100, description="Maximum MCP connections")
    timeout: int = Field(30, description="MCP operation timeout")
    
    class Config:
        env_prefix = "MCP_"


class SystemSettings(BaseSettings):
    """Main system configuration"""
    # LLM settings
    llm: LLMSettings = Field(default_factory=LLMSettings)
    
    # Embedding settings
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    
    # Database settings
    vector_db: VectorDBSettings = Field(default_factory=VectorDBSettings)
    graph_db: GraphDBSettings = Field(default_factory=GraphDBSettings)
    
    # Agent settings
    agent: AgentSettings = Field(default_factory=AgentSettings)
    
    # MCP settings
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    
    # System settings
    log_level: str = Field("INFO", description="Logging level")
    data_dir: str = Field("data", description="Data directory")
    temp_dir: str = Field("/tmp/rag_system", description="Temporary directory")
    
    # API settings
    api_host: str = Field("0.0.0.0", description="API host")
    api_port: int = Field(8080, description="API port")
    
    # Security
    enable_auth: bool = Field(True, description="Enable authentication")
    secret_key: str = Field("your-secret-key-change-this", description="JWT secret key")
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        extra = "allow"


# Global settings instance
settings = SystemSettings()


def get_settings() -> SystemSettings:
    """Get the current system settings"""
    return settings


def update_settings(**kwargs) -> SystemSettings:
    """Update system settings"""
    global settings
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    return settings