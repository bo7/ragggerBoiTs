"""
Vector Store Integration
=======================

Milvus vector database integration for embeddings storage and similarity search.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np

from ..config import get_settings

logger = logging.getLogger(__name__)


class MilvusVectorStore:
    """Milvus vector database client for RAG system"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 19530,
                 user: str = "",
                 password: str = "",
                 secure: bool = False):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.secure = secure
        
        # Milvus client will be initialized lazily
        self.client = None
        self.is_connected = False
        
        # Collection configurations
        self.collections = {}
        
    async def initialize(self) -> bool:
        """Initialize connection to Milvus"""
        try:
            # Import Milvus client
            from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
            
            # Connect to Milvus
            alias = "default"
            connections.connect(
                alias=alias,
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                secure=self.secure
            )
            
            self.is_connected = True
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            
            # Store Milvus modules for later use
            self.connections = connections
            self.Collection = Collection
            self.CollectionSchema = CollectionSchema
            self.FieldSchema = FieldSchema
            self.DataType = DataType
            self.utility = utility
            
            return True
            
        except ImportError:
            logger.error("pymilvus not installed. Install with: pip install pymilvus")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            return False
    
    async def create_collection(self, 
                              collection_name: str,
                              dimension: int = 2048,
                              description: str = "",
                              index_type: str = "IVF_FLAT",
                              metric_type: str = "COSINE") -> bool:
        """
        Create a new collection in Milvus
        
        Args:
            collection_name: Name of the collection
            dimension: Vector dimension (default 2048 for Jina v4)
            description: Collection description
            index_type: Index type for similarity search
            metric_type: Distance metric (COSINE, L2, IP)
            
        Returns:
            Success status
        """
        try:
            if not self.is_connected:
                logger.error("Not connected to Milvus")
                return False
            
            # Check if collection already exists
            if self.utility.has_collection(collection_name):
                logger.info(f"Collection {collection_name} already exists")
                collection = self.Collection(collection_name)
                # Load the existing collection to make it available for search
                collection.load()
                self.collections[collection_name] = collection
                return True
            
            # Define schema
            fields = [
                self.FieldSchema(
                    name="id",
                    dtype=self.DataType.INT64,
                    is_primary=True,
                    auto_id=True
                ),
                self.FieldSchema(
                    name="vector",
                    dtype=self.DataType.FLOAT_VECTOR,
                    dim=dimension
                ),
                self.FieldSchema(
                    name="text",
                    dtype=self.DataType.VARCHAR,
                    max_length=65535
                ),
                self.FieldSchema(
                    name="metadata",
                    dtype=self.DataType.JSON
                ),
                self.FieldSchema(
                    name="source_database",
                    dtype=self.DataType.VARCHAR,
                    max_length=255
                ),
                self.FieldSchema(
                    name="source_schema",
                    dtype=self.DataType.VARCHAR,
                    max_length=255
                ),
                self.FieldSchema(
                    name="source_table",
                    dtype=self.DataType.VARCHAR,
                    max_length=255
                ),
                self.FieldSchema(
                    name="source_id",
                    dtype=self.DataType.VARCHAR,
                    max_length=255
                ),
                self.FieldSchema(
                    name="embedding_model",
                    dtype=self.DataType.VARCHAR,
                    max_length=100
                ),
                self.FieldSchema(
                    name="created_at",
                    dtype=self.DataType.INT64
                )
            ]
            
            schema = self.CollectionSchema(
                fields=fields,
                description=description
            )
            
            # Create collection
            collection = self.Collection(
                name=collection_name,
                schema=schema
            )
            
            # Create index
            index_params = {
                "metric_type": metric_type,
                "index_type": index_type,
                "params": {"nlist": 1024}
            }
            
            collection.create_index(
                field_name="vector",
                index_params=index_params
            )
            
            # Load collection
            collection.load()
            
            self.collections[collection_name] = collection
            logger.info(f"Created and indexed collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            return False
    
    async def insert_vectors(self, 
                           collection_name: str,
                           vectors: List[List[float]],
                           texts: List[str],
                           metadata: List[Dict],
                           source_database: str = "",
                           source_schema: str = "",
                           source_table: str = "",
                           source_ids: List[str] = None,
                           embedding_model: str = "jina-embeddings-v4") -> bool:
        """
        Insert vectors into a collection
        
        Args:
            collection_name: Target collection
            vectors: List of embedding vectors
            texts: Corresponding text content
            metadata: Associated metadata for each vector
            source_table: Source database table
            source_ids: Source record IDs
            embedding_model: Model used for embeddings
            
        Returns:
            Success status
        """
        try:
            if collection_name not in self.collections:
                logger.error(f"Collection {collection_name} not found")
                return False
            
            collection = self.collections[collection_name]
            
            # Prepare data
            import time
            current_time = int(time.time() * 1000)  # milliseconds
            
            if source_ids is None:
                source_ids = [f"{source_database}.{source_schema}.{source_table}_{i}" for i in range(len(vectors))]
            
            data = [
                vectors,  # vector field
                texts,    # text field
                metadata, # metadata field (JSON)
                [source_database] * len(vectors),  # source_database field
                [source_schema] * len(vectors),   # source_schema field
                [source_table] * len(vectors),    # source_table field
                source_ids,  # source_id field
                [embedding_model] * len(vectors),  # embedding_model field
                [current_time] * len(vectors)     # created_at field
            ]
            
            # Insert data
            insert_result = collection.insert(data)
            collection.flush()
            
            logger.info(f"Inserted {len(vectors)} vectors into {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert vectors into {collection_name}: {e}")
            return False
    
    async def search_similar(self, 
                           collection_name: str,
                           query_vector: List[float],
                           limit: int = 10,
                           search_params: Dict = None,
                           filter_expr: str = None) -> List[Dict]:
        """
        Search for similar vectors
        
        Args:
            collection_name: Collection to search in
            query_vector: Query embedding vector
            limit: Number of results to return
            search_params: Search parameters
            filter_expr: Filter expression
            
        Returns:
            List of similar documents with scores
        """
        try:
            if collection_name not in self.collections:
                logger.error(f"Collection {collection_name} not found")
                return []
            
            collection = self.collections[collection_name]
            
            # Default search params
            if search_params is None:
                search_params = {"nprobe": 10}
            
            # Perform search
            search_results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=limit,
                expr=filter_expr,
                output_fields=["text", "metadata", "source_database", "source_schema", "source_table", "source_id", "embedding_model", "created_at"]
            )
            
            # Process results
            results = []
            for hits in search_results:
                for hit in hits:
                    result = {
                        "id": hit.id,
                        "score": hit.score,
                        "text": hit.entity.get("text", ""),
                        "metadata": hit.entity.get("metadata", {}),
                        "source_database": hit.entity.get("source_database", ""),
                        "source_schema": hit.entity.get("source_schema", ""),
                        "source_table": hit.entity.get("source_table", ""),
                        "source_id": hit.entity.get("source_id", ""),
                        "embedding_model": hit.entity.get("embedding_model", ""),
                        "created_at": hit.entity.get("created_at", 0)
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed in {collection_name}: {e}")
            return []
    
    async def hybrid_search(self, 
                          collection_name: str,
                          query_vector: List[float],
                          text_query: str = "",
                          metadata_filter: Dict = None,
                          limit: int = 10) -> List[Dict]:
        """
        Perform hybrid search combining vector similarity and metadata filtering
        
        Args:
            collection_name: Collection to search in
            query_vector: Query embedding vector
            text_query: Text query for additional filtering
            metadata_filter: Metadata-based filters
            limit: Number of results to return
            
        Returns:
            Filtered and ranked results
        """
        try:
            # Build filter expression
            filter_parts = []
            
            if text_query:
                # Simple text matching (can be enhanced with full-text search)
                filter_parts.append(f"text like '%{text_query}%'")
            
            if metadata_filter:
                for key, value in metadata_filter.items():
                    if isinstance(value, str):
                        filter_parts.append(f"metadata['{key}'] == '{value}'")
                    else:
                        filter_parts.append(f"metadata['{key}'] == {value}")
            
            filter_expr = " and ".join(filter_parts) if filter_parts else None
            
            # Perform vector search with filters
            results = await self.search_similar(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                filter_expr=filter_expr
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    async def get_collection_stats(self, collection_name: str) -> Dict:
        """Get statistics for a collection"""
        try:
            if collection_name not in self.collections:
                return {"error": f"Collection {collection_name} not found"}
            
            collection = self.collections[collection_name]
            
            stats = {
                "name": collection_name,
                "num_entities": collection.num_entities,
                "description": collection.description,
                "schema": {
                    "fields": [
                        {
                            "name": field.name,
                            "type": str(field.dtype),
                            "params": field.params
                        }
                        for field in collection.schema.fields
                    ]
                }
            }
            
            # Get index information
            try:
                index_info = collection.index()
                stats["index"] = {
                    "metric_type": index_info.params.get("metric_type", "unknown"),
                    "index_type": index_info.params.get("index_type", "unknown")
                }
            except:
                stats["index"] = {"error": "Could not retrieve index info"}
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            if collection_name in self.collections:
                self.collections[collection_name].drop()
                del self.collections[collection_name]
                logger.info(f"Deleted collection: {collection_name}")
                return True
            else:
                logger.warning(f"Collection {collection_name} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return False
    
    async def list_collections(self) -> List[str]:
        """List all collections"""
        try:
            if not self.is_connected:
                return []
            
            return self.utility.list_collections()
            
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    async def health_check(self) -> Dict:
        """Check Milvus connection health"""
        try:
            if not self.is_connected:
                return {"healthy": False, "error": "Not connected"}
            
            # Try to list collections as a health check
            collections = await self.list_collections()
            
            return {
                "healthy": True,
                "host": self.host,
                "port": self.port,
                "collections": collections,
                "num_collections": len(collections)
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def shutdown(self):
        """Close Milvus connection"""
        try:
            if self.is_connected:
                self.connections.disconnect("default")
                self.is_connected = False
                logger.info("Disconnected from Milvus")
                
        except Exception as e:
            logger.error(f"Error during Milvus shutdown: {e}")


class VectorStoreManager:
    """High-level manager for vector store operations"""
    
    def __init__(self, vector_store: MilvusVectorStore):
        self.vector_store = vector_store
        
    async def setup_rag_collections(self) -> bool:
        """Set up standard collections for RAG system"""
        try:
            collections_to_create = [
                {
                    "collection_name": "documents",
                    "description": "Main document collection for RAG",
                    "dimension": 2048
                },
                {
                    "collection_name": "code_snippets", 
                    "description": "Code snippets and technical documentation",
                    "dimension": 2048
                },
                {
                    "collection_name": "database_schema",
                    "description": "Database schema and metadata",
                    "dimension": 2048
                }
            ]
            
            for collection_config in collections_to_create:
                success = await self.vector_store.create_collection(**collection_config)
                if not success:
                    logger.error(f"Failed to create collection: {collection_config['collection_name']}")
                    return False
            
            logger.info("Successfully set up all RAG collections")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup RAG collections: {e}")
            return False
    
    async def ingest_sql_data(self, 
                            table_data: List[Dict],
                            embeddings: List[List[float]],
                            database_name: str,
                            schema_name: str,
                            table_name: str,
                            collection_name: str = "documents") -> bool:
        """
        Ingest SQL Server data with embeddings into vector store
        
        Args:
            table_data: List of database records
            embeddings: Corresponding embeddings for each record
            table_name: Source table name
            collection_name: Target collection
            
        Returns:
            Success status
        """
        try:
            # Prepare data for vector store
            texts = []
            metadata = []
            source_ids = []
            
            for record in table_data:
                # Convert record to searchable text
                text_parts = []
                clean_metadata = {}
                
                for key, value in record.items():
                    if value is not None:
                        text_parts.append(f"{key}: {value}")
                        clean_metadata[key] = value
                
                texts.append(" | ".join(text_parts))
                metadata.append(clean_metadata)
                source_ids.append(str(record.get('id', record.get('ID', len(source_ids)))))
            
            # Insert into vector store
            success = await self.vector_store.insert_vectors(
                collection_name=collection_name,
                vectors=embeddings,
                texts=texts,
                metadata=metadata,
                source_database=database_name,
                source_schema=schema_name,
                source_table=table_name,
                source_ids=source_ids
            )
            
            if success:
                logger.info(f"Successfully ingested {len(table_data)} records from {database_name}.{schema_name}.{table_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to ingest SQL data: {e}")
            return False