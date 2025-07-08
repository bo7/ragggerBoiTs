"""
RAG Query Engine
===============

Unified query engine that orchestrates vector search, graph traversal,
and LLM response generation for the multi-database RAG system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from .intelligent_router import IntelligentRouter, QueryClassification, QueryType
from ..llm_serving.llm_client import LLMClient
from ..embedding.jina_client import JinaEmbeddingClient
from ..storage.vector_store import MilvusVectorStore
from ..storage.graph_store import Neo4jGraphStore
from ..config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from RAG query engine"""
    query: str
    answer: str
    query_type: str
    confidence: float
    sources: List[Dict[str, Any]]
    execution_time: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "query": self.query,
            "answer": self.answer,
            "query_type": self.query_type,
            "confidence": self.confidence,
            "sources": self.sources,
            "execution_time": self.execution_time,
            "metadata": self.metadata
        }


class RAGEngine:
    """Unified RAG query engine"""
    
    def __init__(self,
                 router: IntelligentRouter = None,
                 llm_client: LLMClient = None,
                 embedding_client: JinaEmbeddingClient = None,
                 vector_store: MilvusVectorStore = None,
                 graph_store: Neo4jGraphStore = None):
        
        # Initialize components
        self.router = router or IntelligentRouter()
        self.llm_client = llm_client or LLMClient()
        self.embedding_client = embedding_client or JinaEmbeddingClient()
        self.vector_store = vector_store or MilvusVectorStore()
        self.graph_store = graph_store or Neo4jGraphStore()
        
        self.is_initialized = False
        
        # Response generation prompts
        self.vector_response_prompt = self._build_vector_response_prompt()
        self.graph_response_prompt = self._build_graph_response_prompt()
        self.hybrid_response_prompt = self._build_hybrid_response_prompt()
        
    async def initialize(self) -> bool:
        """Initialize all components"""
        try:
            logger.info("Initializing RAG engine components...")
            
            # Initialize router
            if not await self.router.initialize():
                logger.error("Failed to initialize query router")
                return False
            
            # Initialize LLM client
            if not await self.llm_client.initialize():
                logger.error("Failed to initialize LLM client")
                return False
            
            # Initialize embedding client
            if not await self.embedding_client.initialize():
                logger.error("Failed to initialize embedding client")
                return False
            
            # Initialize vector store
            if not await self.vector_store.initialize():
                logger.error("Failed to initialize vector store")
                return False
            
            # Initialize graph store
            if not await self.graph_store.initialize():
                logger.error("Failed to initialize graph store")
                return False
            
            self.is_initialized = True
            logger.info("RAG engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}")
            return False
    
    def _build_vector_response_prompt(self) -> str:
        """Build prompt for vector-based responses"""
        return """
You are a helpful AI assistant answering questions based on retrieved data from a multi-database system.

User Query: {query}

Retrieved Data:
{retrieved_data}

Instructions:
1. Answer the user's question based on the retrieved data
2. Be specific and cite relevant information from the sources
3. If the data is incomplete, mention what additional information would be helpful
4. Format your response clearly and concisely
5. Include relevant details like database sources when helpful

Answer:"""
    
    def _build_graph_response_prompt(self) -> str:
        """Build prompt for graph-based responses"""
        return """
You are a helpful AI assistant answering questions about database relationships and structure.

User Query: {query}

Graph Data:
{graph_data}

Instructions:
1. Answer the user's question based on the graph relationship data
2. Explain the relationships clearly and logically
3. Use specific examples from the data when possible
4. If asking about structure, provide a clear overview
5. Mention relevant database, schema, and table names

Answer:"""
    
    def _build_hybrid_response_prompt(self) -> str:
        """Build prompt for hybrid responses"""
        return """
You are a helpful AI assistant answering complex questions using both content data and relationship information.

User Query: {query}

Content Data:
{content_data}

Relationship Data:
{relationship_data}

Instructions:
1. Synthesize information from both content and relationship data
2. Provide a comprehensive answer that leverages both types of information
3. Show how the relationships connect to the content
4. Be specific and cite sources appropriately
5. Structure your response logically

Answer:"""
    
    async def query(self, query: str) -> RAGResponse:
        """Process a query through the RAG pipeline"""
        start_time = datetime.now()
        
        try:
            if not self.is_initialized:
                raise RuntimeError("RAG engine not initialized")
            
            # Route the query
            classification = await self.router.route_query(query)
            
            # Execute query based on classification
            if classification.query_type == QueryType.VECTOR:
                answer, sources = await self._execute_vector_query(query, classification)
            elif classification.query_type == QueryType.GRAPH:
                answer, sources = await self._execute_graph_query(query, classification)
            elif classification.query_type == QueryType.HYBRID:
                answer, sources = await self._execute_hybrid_query(query, classification)
            else:
                raise ValueError(f"Unknown query type: {classification.query_type}")
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Build response
            response = RAGResponse(
                query=query,
                answer=answer,
                query_type=classification.query_type.value,
                confidence=classification.confidence,
                sources=sources,
                execution_time=execution_time,
                metadata={
                    "classification": classification.to_dict(),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            logger.info(f"Query processed in {execution_time:.2f}s using {classification.query_type.value} approach")
            return response
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Query processing failed: {e}")
            
            # Return error response
            return RAGResponse(
                query=query,
                answer=f"Sorry, I encountered an error processing your query: {e}",
                query_type="error",
                confidence=0.0,
                sources=[],
                execution_time=execution_time,
                metadata={"error": str(e)}
            )
    
    async def _execute_vector_query(self, query: str, classification: QueryClassification) -> Tuple[str, List[Dict]]:
        """Execute vector similarity search"""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_client.embed_text(
                [query],
                task="retrieval.query"
            )
            
            if not query_embedding:
                raise Exception("Failed to generate query embedding")
            
            # Search vector store
            search_results = await self.vector_store.search_similar(
                collection_name="documents",
                query_vector=query_embedding[0],
                limit=10
            )
            
            if not search_results:
                return "I couldn't find any relevant information for your query.", []
            
            # Format retrieved data
            retrieved_data = self._format_vector_results(search_results)
            
            # Generate response
            prompt = self.vector_response_prompt.format(
                query=query,
                retrieved_data=retrieved_data
            )
            
            answer = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.3
            )
            
            # Format sources
            sources = self._format_vector_sources(search_results)
            
            return answer, sources
            
        except Exception as e:
            logger.error(f"Vector query execution failed: {e}")
            return f"Failed to execute vector search: {e}", []
    
    async def _execute_graph_query(self, query: str, classification: QueryClassification) -> Tuple[str, List[Dict]]:
        """Execute graph traversal query"""
        try:
            # Determine graph query based on patterns
            graph_query = self._build_graph_query(query, classification)
            
            # Execute graph query
            graph_results = await self.graph_store.execute_query(graph_query)
            
            if not graph_results:
                return "I couldn't find any relevant relationship information for your query.", []
            
            # Format graph data
            graph_data = self._format_graph_results(graph_results)
            
            # Generate response
            prompt = self.graph_response_prompt.format(
                query=query,
                graph_data=graph_data
            )
            
            answer = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.3
            )
            
            # Format sources
            sources = self._format_graph_sources(graph_results)
            
            return answer, sources
            
        except Exception as e:
            logger.error(f"Graph query execution failed: {e}")
            return f"Failed to execute graph query: {e}", []
    
    async def _execute_hybrid_query(self, query: str, classification: QueryClassification) -> Tuple[str, List[Dict]]:
        """Execute hybrid vector + graph query"""
        try:
            # Execute both vector and graph queries
            vector_task = self._execute_vector_query(query, classification)
            graph_task = self._execute_graph_query(query, classification)
            
            # Run concurrently
            (vector_answer, vector_sources), (graph_answer, graph_sources) = await asyncio.gather(
                vector_task, graph_task
            )
            
            # Combine results
            content_data = self._extract_content_from_answer(vector_answer, vector_sources)
            relationship_data = self._extract_relationships_from_answer(graph_answer, graph_sources)
            
            # Generate hybrid response
            prompt = self.hybrid_response_prompt.format(
                query=query,
                content_data=content_data,
                relationship_data=relationship_data
            )
            
            answer = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=1200,
                temperature=0.3
            )
            
            # Combine sources
            sources = vector_sources + graph_sources
            
            return answer, sources
            
        except Exception as e:
            logger.error(f"Hybrid query execution failed: {e}")
            return f"Failed to execute hybrid query: {e}", []
    
    def _build_graph_query(self, query: str, classification: QueryClassification) -> str:
        """Build Cypher query based on classification"""
        query_lower = query.lower()
        
        # Default query for database structure
        if any(word in query_lower for word in ["schema", "structure", "database", "table"]):
            return """
            MATCH (db:Database)-[:CONTAINS]->(schema:Schema)-[:CONTAINS]->(table:Table)
            RETURN db.database as database, schema.schema as schema, table.table_name as table_name, table.row_count as row_count
            ORDER BY db.database, schema.schema, table.table_name
            LIMIT 20
            """
        
        # Query for relationships
        if any(word in query_lower for word in ["relationship", "related", "connection"]):
            return """
            MATCH (from:Table)-[r:REFERENCES]->(to:Table)
            RETURN from.database as from_database, from.schema as from_schema, from.table_name as from_table,
                   to.database as to_database, to.schema as to_schema, to.table_name as to_table,
                   r.constraint_name as constraint_name
            LIMIT 20
            """
        
        # Default: show database overview
        return """
        MATCH (db:Database)
        OPTIONAL MATCH (db)-[:CONTAINS]->(schema:Schema)
        OPTIONAL MATCH (schema)-[:CONTAINS]->(table:Table)
        RETURN db.database as database, count(DISTINCT schema) as schema_count, count(table) as table_count
        ORDER BY db.database
        """
    
    def _format_vector_results(self, results: List[Dict]) -> str:
        """Format vector search results for prompt"""
        formatted_results = []
        
        for i, result in enumerate(results[:5], 1):  # Limit to top 5
            formatted_results.append(f"""
Result {i} (Score: {result.get('score', 0):.3f}):
Database: {result.get('source_database', 'Unknown')}
Schema: {result.get('source_schema', 'Unknown')}
Table: {result.get('source_table', 'Unknown')}
Content: {result.get('text', 'No content')[:200]}...
""")
        
        return "\\n".join(formatted_results)
    
    def _format_graph_results(self, results: List[Dict]) -> str:
        """Format graph query results for prompt"""
        formatted_results = []
        
        for i, result in enumerate(results[:10], 1):  # Limit to top 10
            formatted_results.append(f"Result {i}: {json.dumps(result, indent=2)}")
        
        return "\\n".join(formatted_results)
    
    def _format_vector_sources(self, results: List[Dict]) -> List[Dict]:
        """Format vector sources for response"""
        sources = []
        
        for result in results:
            source = {
                "type": "vector",
                "database": result.get('source_database', 'Unknown'),
                "schema": result.get('source_schema', 'Unknown'),
                "table": result.get('source_table', 'Unknown'),
                "score": result.get('score', 0),
                "content_preview": result.get('text', '')[:100] + "..." if result.get('text') else ""
            }
            sources.append(source)
        
        return sources
    
    def _format_graph_sources(self, results: List[Dict]) -> List[Dict]:
        """Format graph sources for response"""
        sources = []
        
        for result in results:
            source = {
                "type": "graph",
                "data": result
            }
            sources.append(source)
        
        return sources
    
    def _extract_content_from_answer(self, answer: str, sources: List[Dict]) -> str:
        """Extract content information from vector answer"""
        return f"Content Answer: {answer}\\n\\nSources: {len(sources)} vector results"
    
    def _extract_relationships_from_answer(self, answer: str, sources: List[Dict]) -> str:
        """Extract relationship information from graph answer"""
        return f"Relationship Answer: {answer}\\n\\nSources: {len(sources)} graph results"
    
    async def get_engine_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        try:
            stats = {
                "is_initialized": self.is_initialized,
                "components": {}
            }
            
            # Get component health
            if self.is_initialized:
                stats["components"]["router"] = await self.router.get_router_stats()
                stats["components"]["llm"] = await self.llm_client.health_check()
                stats["components"]["embedding"] = await self.embedding_client.health_check()
                stats["components"]["vector_store"] = await self.vector_store.health_check()
                stats["components"]["graph_store"] = await self.graph_store.health_check()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get engine stats: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Shutdown all components"""
        try:
            logger.info("Shutting down RAG engine...")
            
            # Shutdown components
            if self.router:
                await self.router.shutdown()
            
            if self.llm_client:
                await self.llm_client.shutdown()
            
            if self.embedding_client:
                await self.embedding_client.shutdown()
            
            if self.vector_store:
                await self.vector_store.shutdown()
            
            if self.graph_store:
                await self.graph_store.shutdown()
            
            self.is_initialized = False
            logger.info("RAG engine shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during RAG engine shutdown: {e}")


async def main():
    """Example usage of RAG engine"""
    engine = RAGEngine()
    
    try:
        # Initialize engine
        if not await engine.initialize():
            logger.error("Failed to initialize RAG engine")
            return
        
        # Test queries
        test_queries = [
            "Show me information about customers",
            "What is the database schema structure?",
            "Find high-value customers and their relationship to products"
        ]
        
        for query in test_queries:
            print(f"\\n{'='*50}")
            print(f"Query: {query}")
            print(f"{'='*50}")
            
            response = await engine.query(query)
            
            print(f"Answer: {response.answer}")
            print(f"Query Type: {response.query_type}")
            print(f"Confidence: {response.confidence:.2f}")
            print(f"Execution Time: {response.execution_time:.2f}s")
            print(f"Sources: {len(response.sources)}")
        
        # Get stats
        stats = await engine.get_engine_stats()
        print(f"\\nEngine Stats: {json.dumps(stats, indent=2)}")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        
    finally:
        await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())