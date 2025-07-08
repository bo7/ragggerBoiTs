"""
RAG Data Pipeline
================

Orchestrates the complete data pipeline from SQL Server through dbt semantic layer
to vector and graph databases, creating business-aware embeddings.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .dbt_integration import DBTSemanticExtractor
from .sql_extractor import EnhancedSQLExtractor
from ..embedding import JinaEmbeddingClient
from ..storage import MilvusVectorStore, VectorStoreManager

logger = logging.getLogger(__name__)


class RAGDataPipeline:
    """Complete RAG data pipeline with dbt semantic integration"""
    
    def __init__(self,
                 dbt_project_dir: Path = None,
                 output_dir: Path = None,
                 vector_store: MilvusVectorStore = None,
                 embedding_client: JinaEmbeddingClient = None):
        
        self.dbt_project_dir = dbt_project_dir or Path.cwd()
        self.output_dir = output_dir or Path.cwd() / "rag_output"
        self.output_dir.mkdir(exist_ok=True)
        
        # Components
        self.dbt_extractor = DBTSemanticExtractor(dbt_project_dir)
        self.sql_extractor = None
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.vector_manager = None
        
        # Pipeline state
        self.pipeline_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.extracted_data = {}
        self.embeddings_cache = {}
        
    async def initialize(self) -> bool:
        """Initialize all pipeline components"""
        try:
            logger.info(f"Initializing RAG pipeline {self.pipeline_id}")
            
            # Initialize dbt extractor
            if not await self.dbt_extractor.initialize():
                logger.error("Failed to initialize dbt extractor")
                return False
            
            # Initialize SQL extractor with dbt integration
            self.sql_extractor = EnhancedSQLExtractor(
                dbt_extractor=self.dbt_extractor,
                include_constraints=True,
                include_relationships=True
            )
            
            if not await self.sql_extractor.initialize():
                logger.error("Failed to initialize SQL extractor")
                return False
            
            # Initialize embedding client
            if self.embedding_client and not self.embedding_client.is_initialized:
                if not await self.embedding_client.initialize():
                    logger.error("Failed to initialize embedding client")
                    return False
            
            # Initialize vector store
            if self.vector_store and not self.vector_store.is_connected:
                if not await self.vector_store.initialize():
                    logger.error("Failed to initialize vector store")
                    return False
                
                self.vector_manager = VectorStoreManager(self.vector_store)
            
            logger.info("RAG pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            return False
    
    async def run_full_pipeline(self,
                              extract_data: bool = True,
                              generate_embeddings: bool = True,
                              store_vectors: bool = True,
                              store_graph: bool = False) -> Dict[str, Any]:
        """Run the complete RAG pipeline"""
        try:
            logger.info(f"Starting full RAG pipeline execution")
            
            pipeline_results = {
                "pipeline_id": self.pipeline_id,
                "start_time": datetime.now().isoformat(),
                "steps_completed": [],
                "stats": {},
                "errors": []
            }
            
            # Step 1: Extract enhanced data
            if extract_data:
                logger.info("Step 1: Extracting enhanced data from SQL Server")
                try:
                    self.extracted_data = await self.sql_extractor.extract_enhanced_catalog()
                    
                    # Save extracted data
                    extract_file = self.output_dir / f"enhanced_catalog_{self.pipeline_id}.json"
                    with open(extract_file, 'w', encoding='utf-8') as f:
                        json.dump(self.extracted_data, f, indent=2, default=str)
                    
                    pipeline_results["steps_completed"].append("data_extraction")
                    pipeline_results["stats"]["tables_extracted"] = len(self.extracted_data.get("tables", []))
                    
                    logger.info(f"Extracted {len(self.extracted_data.get('tables', []))} tables")
                    
                except Exception as e:
                    error_msg = f"Data extraction failed: {e}"
                    logger.error(error_msg)
                    pipeline_results["errors"].append(error_msg)
                    return pipeline_results
            
            # Step 2: Generate embeddings
            if generate_embeddings and self.embedding_client:
                logger.info("Step 2: Generating business-aware embeddings")
                try:
                    embeddings_results = await self._generate_all_embeddings()
                    
                    pipeline_results["steps_completed"].append("embedding_generation")
                    pipeline_results["stats"]["embeddings_generated"] = embeddings_results["total_embeddings"]
                    pipeline_results["stats"]["embedding_dimension"] = embeddings_results["dimension"]
                    
                    logger.info(f"Generated {embeddings_results['total_embeddings']} embeddings")
                    
                except Exception as e:
                    error_msg = f"Embedding generation failed: {e}"
                    logger.error(error_msg)
                    pipeline_results["errors"].append(error_msg)
            
            # Step 3: Store in vector database
            if store_vectors and self.vector_store:
                logger.info("Step 3: Storing vectors in Milvus")
                try:
                    vector_results = await self._store_vectors()
                    
                    pipeline_results["steps_completed"].append("vector_storage")
                    pipeline_results["stats"]["vectors_stored"] = vector_results["vectors_stored"]
                    pipeline_results["stats"]["collections_created"] = vector_results["collections_created"]
                    
                    logger.info(f"Stored {vector_results['vectors_stored']} vectors")
                    
                except Exception as e:
                    error_msg = f"Vector storage failed: {e}"
                    logger.error(error_msg)
                    pipeline_results["errors"].append(error_msg)
            
            # Step 4: Store in graph database (optional)
            if store_graph:
                logger.info("Step 4: Storing relationships in Neo4j")
                try:
                    graph_results = await self._store_graph_relationships()
                    
                    pipeline_results["steps_completed"].append("graph_storage")
                    pipeline_results["stats"]["graph_nodes"] = graph_results["nodes_created"]
                    pipeline_results["stats"]["graph_relationships"] = graph_results["relationships_created"]
                    
                    logger.info(f"Created {graph_results['nodes_created']} nodes and {graph_results['relationships_created']} relationships")
                    
                except Exception as e:
                    error_msg = f"Graph storage failed: {e}"
                    logger.error(error_msg)
                    pipeline_results["errors"].append(error_msg)
            
            pipeline_results["end_time"] = datetime.now().isoformat()
            pipeline_results["success"] = len(pipeline_results["errors"]) == 0
            
            # Save pipeline results
            results_file = self.output_dir / f"pipeline_results_{self.pipeline_id}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(pipeline_results, f, indent=2, default=str)
            
            logger.info(f"Pipeline completed with {len(pipeline_results['errors'])} errors")
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise
    
    async def _generate_all_embeddings(self) -> Dict[str, Any]:
        """Generate embeddings for all extracted tables"""
        try:
            embeddings_results = {
                "total_embeddings": 0,
                "dimension": 0,
                "embeddings_by_table": {},
                "semantic_embeddings": {},
                "constraint_embeddings": {}
            }
            
            tables = self.extracted_data.get("tables", [])
            
            for table_info in tables:
                schema = table_info["schema"]
                table_name = table_info["name"]
                table_key = f"{schema}.{table_name}"
                
                logger.info(f"Generating embeddings for {table_key}")
                
                # Generate different types of embeddings
                table_embeddings = await self._generate_table_embeddings(table_info)
                embeddings_results["embeddings_by_table"][table_key] = table_embeddings
                
                # Count total embeddings
                for embedding_type, embeddings in table_embeddings.items():
                    if embeddings:
                        embeddings_results["total_embeddings"] += len(embeddings)
                        if embeddings_results["dimension"] == 0 and embeddings[0]:
                            embeddings_results["dimension"] = len(embeddings[0])
            
            # Cache embeddings for vector storage
            self.embeddings_cache = embeddings_results
            
            # Save embeddings
            embeddings_file = self.output_dir / f"embeddings_{self.pipeline_id}.json"
            with open(embeddings_file, 'w', encoding='utf-8') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_safe_embeddings = self._make_json_safe(embeddings_results)
                json.dump(json_safe_embeddings, f, indent=2)
            
            return embeddings_results
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    async def _generate_table_embeddings(self, table_info: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        """Generate multiple types of embeddings for a table"""
        try:
            embeddings = {
                "table_description": [],
                "column_descriptions": [],
                "sample_data": [],
                "constraints": [],
                "semantic_context": []
            }
            
            schema = table_info["schema"]
            table_name = table_info["name"]
            
            # 1. Table-level semantic embedding
            if "semantic_description" in table_info and table_info["semantic_description"]:
                table_desc_embedding = await self.embedding_client.embed_text(
                    table_info["semantic_description"],
                    task="retrieval.passage"
                )
                embeddings["table_description"] = table_desc_embedding
            
            # 2. Column-level embeddings
            columns = table_info.get("columns", [])
            if columns:
                column_texts = []
                
                for col in columns:
                    col_text_parts = [f"Column: {col['name']} ({col['data_type']})"]
                    
                    if col.get("description"):
                        col_text_parts.append(f"Description: {col['description']}")
                    
                    if col.get("constraints"):
                        col_text_parts.append(f"Constraints: {', '.join(col['constraints'])}")
                    
                    if col.get("is_primary_key"):
                        col_text_parts.append("Primary Key")
                    
                    if col.get("foreign_key_reference"):
                        col_text_parts.append(f"References: {col['foreign_key_reference']}")
                    
                    column_texts.append(" | ".join(col_text_parts))
                
                if column_texts:
                    column_embeddings = await self.embedding_client.embed_text(
                        column_texts,
                        task="retrieval.passage"
                    )
                    embeddings["column_descriptions"] = column_embeddings
            
            # 3. Sample data embeddings
            sample_rows = table_info.get("sample_rows", [])
            if sample_rows:
                sample_texts = []
                
                for row in sample_rows[:3]:  # Limit to first 3 samples
                    row_text = f"Sample data from {schema}.{table_name}: "
                    row_parts = []
                    
                    for key, value in row.items():
                        if value is not None:
                            row_parts.append(f"{key}={value}")
                    
                    sample_texts.append(row_text + " | ".join(row_parts))
                
                if sample_texts:
                    sample_embeddings = await self.embedding_client.embed_text(
                        sample_texts,
                        task="text-matching"
                    )
                    embeddings["sample_data"] = sample_embeddings
            
            # 4. Business constraints and rules
            constraint_texts = []
            
            # Add database constraints
            constraints_info = self.extracted_data.get("constraints", {}).get(f"{schema}.{table_name}", {})
            if constraints_info:
                if constraints_info.get("primary_keys"):
                    constraint_texts.append(f"Primary keys: {', '.join(constraints_info['primary_keys'])}")
                
                if constraints_info.get("foreign_keys"):
                    fk_texts = [f"{fk['column']} references {fk['references']}" for fk in constraints_info["foreign_keys"]]
                    constraint_texts.append(f"Foreign keys: {'; '.join(fk_texts)}")
                
                if constraints_info.get("dbt_business_rules"):
                    constraint_texts.extend(constraints_info["dbt_business_rules"])
            
            if constraint_texts:
                constraint_embeddings = await self.embedding_client.embed_text(
                    constraint_texts,
                    task="retrieval.passage"
                )
                embeddings["constraints"] = constraint_embeddings
            
            # 5. Semantic context from dbt
            semantic_info = self.extracted_data.get("semantic_info", {}).get(f"{schema}.{table_name}", {})
            if semantic_info:
                context_parts = []
                
                if semantic_info.get("model_type"):
                    context_parts.append(f"dbt model type: {semantic_info['model_type']}")
                
                if semantic_info.get("dependencies"):
                    context_parts.append(f"Depends on: {', '.join(semantic_info['dependencies'])}")
                
                if semantic_info.get("business_rules"):
                    context_parts.extend(semantic_info["business_rules"])
                
                if context_parts:
                    context_text = f"Business context for {schema}.{table_name}: " + " | ".join(context_parts)
                    semantic_embeddings = await self.embedding_client.embed_text(
                        [context_text],
                        task="retrieval.passage"
                    )
                    embeddings["semantic_context"] = semantic_embeddings
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate table embeddings: {e}")
            return {}
    
    async def _store_vectors(self) -> Dict[str, Any]:
        """Store all embeddings in Milvus vector database"""
        try:
            storage_results = {
                "vectors_stored": 0,
                "collections_created": 0,
                "storage_by_collection": {}
            }
            
            # Ensure RAG collections exist
            if not await self.vector_manager.setup_rag_collections():
                raise RuntimeError("Failed to setup RAG collections")
            
            storage_results["collections_created"] = 3  # documents, code_snippets, database_schema
            
            # Store table-level information in documents collection
            for table_key, table_embeddings in self.embeddings_cache["embeddings_by_table"].items():
                schema, table_name = table_key.split(".", 1)
                
                # Find corresponding table info
                table_info = None
                for tbl in self.extracted_data["tables"]:
                    if tbl["schema"] == schema and tbl["name"] == table_name:
                        table_info = tbl
                        break
                
                if not table_info:
                    continue
                
                # Store table description
                if table_embeddings.get("table_description"):
                    await self._store_table_vectors(
                        "documents",
                        table_info,
                        table_embeddings["table_description"],
                        "table_description"
                    )
                    storage_results["vectors_stored"] += len(table_embeddings["table_description"])
                
                # Store column descriptions
                if table_embeddings.get("column_descriptions"):
                    await self._store_column_vectors(
                        "database_schema",
                        table_info,
                        table_embeddings["column_descriptions"]
                    )
                    storage_results["vectors_stored"] += len(table_embeddings["column_descriptions"])
                
                # Store constraint information
                if table_embeddings.get("constraints"):
                    await self._store_constraint_vectors(
                        "database_schema",
                        table_info,
                        table_embeddings["constraints"]
                    )
                    storage_results["vectors_stored"] += len(table_embeddings["constraints"])
                
                # Store semantic context
                if table_embeddings.get("semantic_context"):
                    await self._store_semantic_vectors(
                        "documents",
                        table_info,
                        table_embeddings["semantic_context"]
                    )
                    storage_results["vectors_stored"] += len(table_embeddings["semantic_context"])
            
            return storage_results
            
        except Exception as e:
            logger.error(f"Failed to store vectors: {e}")
            raise
    
    async def _store_table_vectors(self, collection: str, table_info: Dict, embeddings: List[List[float]], vector_type: str):
        """Store table-level vectors"""
        try:
            texts = [table_info.get("semantic_description", f"Table: {table_info['schema']}.{table_info['name']}")]
            metadata = [{
                "table_schema": table_info["schema"],
                "table_name": table_info["name"],
                "vector_type": vector_type,
                "column_count": table_info.get("column_count", 0),
                "row_count": table_info.get("statistics", {}).get("row_count", "unknown"),
                "dbt_model_type": table_info.get("dbt_model_type", "unknown")
            }]
            
            await self.vector_store.insert_vectors(
                collection_name=collection,
                vectors=embeddings,
                texts=texts,
                metadata=metadata,
                source_table=f"{table_info['schema']}.{table_info['name']}",
                source_ids=[f"{table_info['schema']}.{table_info['name']}.table_desc"]
            )
            
        except Exception as e:
            logger.error(f"Failed to store table vectors: {e}")
    
    async def _store_column_vectors(self, collection: str, table_info: Dict, embeddings: List[List[float]]):
        """Store column-level vectors"""
        try:
            columns = table_info.get("columns", [])
            if len(embeddings) != len(columns):
                logger.warning(f"Mismatch between embeddings ({len(embeddings)}) and columns ({len(columns)})")
                return
            
            texts = []
            metadata = []
            source_ids = []
            
            for i, col in enumerate(columns):
                col_text = f"Column {col['name']} in {table_info['schema']}.{table_info['name']}: {col['data_type']}"
                if col.get("description"):
                    col_text += f" - {col['description']}"
                
                texts.append(col_text)
                metadata.append({
                    "table_schema": table_info["schema"],
                    "table_name": table_info["name"],
                    "column_name": col["name"],
                    "data_type": col["data_type"],
                    "is_nullable": col.get("is_nullable", True),
                    "is_primary_key": col.get("is_primary_key", False),
                    "is_foreign_key": col.get("is_foreign_key", False),
                    "vector_type": "column_description"
                })
                source_ids.append(f"{table_info['schema']}.{table_info['name']}.{col['name']}")
            
            await self.vector_store.insert_vectors(
                collection_name=collection,
                vectors=embeddings,
                texts=texts,
                metadata=metadata,
                source_table=f"{table_info['schema']}.{table_info['name']}",
                source_ids=source_ids
            )
            
        except Exception as e:
            logger.error(f"Failed to store column vectors: {e}")
    
    async def _store_constraint_vectors(self, collection: str, table_info: Dict, embeddings: List[List[float]]):
        """Store constraint-related vectors"""
        try:
            # This would be implemented based on the constraint texts generated
            # For now, we'll create a simple implementation
            texts = [f"Business constraints for {table_info['schema']}.{table_info['name']}"]
            metadata = [{
                "table_schema": table_info["schema"],
                "table_name": table_info["name"],
                "vector_type": "constraints"
            }]
            source_ids = [f"{table_info['schema']}.{table_info['name']}.constraints"]
            
            await self.vector_store.insert_vectors(
                collection_name=collection,
                vectors=embeddings,
                texts=texts,
                metadata=metadata,
                source_table=f"{table_info['schema']}.{table_info['name']}",
                source_ids=source_ids
            )
            
        except Exception as e:
            logger.error(f"Failed to store constraint vectors: {e}")
    
    async def _store_semantic_vectors(self, collection: str, table_info: Dict, embeddings: List[List[float]]):
        """Store semantic context vectors"""
        try:
            texts = [f"Business context for {table_info['schema']}.{table_info['name']}"]
            metadata = [{
                "table_schema": table_info["schema"],
                "table_name": table_info["name"],
                "vector_type": "semantic_context",
                "dbt_model_type": table_info.get("dbt_model_type", "unknown")
            }]
            source_ids = [f"{table_info['schema']}.{table_info['name']}.semantic"]
            
            await self.vector_store.insert_vectors(
                collection_name=collection,
                vectors=embeddings,
                texts=texts,
                metadata=metadata,
                source_table=f"{table_info['schema']}.{table_info['name']}",
                source_ids=source_ids
            )
            
        except Exception as e:
            logger.error(f"Failed to store semantic vectors: {e}")
    
    async def _store_graph_relationships(self) -> Dict[str, Any]:
        """Store relationships in Neo4j graph database"""
        try:
            # This would be implemented with Neo4j client
            # For now, return placeholder results
            return {
                "nodes_created": 0,
                "relationships_created": 0,
                "message": "Graph storage not implemented yet"
            }
            
        except Exception as e:
            logger.error(f"Failed to store graph relationships: {e}")
            raise
    
    def _make_json_safe(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-JSON types to JSON-safe types"""
        if isinstance(obj, dict):
            return {key: self._make_json_safe(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_safe(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        else:
            return obj
    
    async def search_semantic(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search the RAG system using semantic similarity"""
        try:
            if not self.embedding_client or not self.vector_store:
                raise RuntimeError("Embedding client and vector store required for search")
            
            # Generate query embedding
            query_embedding = await self.embedding_client.embed_text(
                query,
                task="retrieval.query"
            )
            
            # Search in all collections
            all_results = []
            
            collections = ["documents", "database_schema", "code_snippets"]
            for collection in collections:
                try:
                    results = await self.vector_store.search_similar(
                        collection_name=collection,
                        query_vector=query_embedding[0],
                        limit=limit
                    )
                    
                    for result in results:
                        result["collection"] = collection
                        all_results.append(result)
                        
                except Exception as e:
                    logger.warning(f"Search failed in collection {collection}: {e}")
            
            # Sort by score and return top results
            all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            return all_results[:limit]
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def cleanup(self):
        """Clean up pipeline resources"""
        try:
            if self.sql_extractor:
                await self.sql_extractor.close()
            
            if self.embedding_client:
                await self.embedding_client.shutdown()
            
            if self.vector_store:
                await self.vector_store.shutdown()
            
            logger.info("Pipeline cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()