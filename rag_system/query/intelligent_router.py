"""
Intelligent Query Router
========================

LLM-powered query routing that decides whether to use vector search,
graph traversal, or hybrid approaches based on query analysis.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import re

from ..llm_serving.llm_client import LLMClient
from ..config import get_settings

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries that can be routed"""
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


@dataclass
class QueryClassification:
    """Result of query classification"""
    query_type: QueryType
    confidence: float
    reasoning: str
    vector_search_terms: List[str]
    graph_patterns: List[str]
    suggested_filters: Dict[str, Any]
    complexity_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "query_type": self.query_type.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "vector_search_terms": self.vector_search_terms,
            "graph_patterns": self.graph_patterns,
            "suggested_filters": self.suggested_filters,
            "complexity_score": self.complexity_score
        }


class IntelligentRouter:
    """LLM-powered query router for RAG system"""
    
    def __init__(self, llm_client: LLMClient = None):
        self.llm_client = llm_client or LLMClient()
        self.is_initialized = False
        
        # Cache for query patterns
        self.pattern_cache = {}
        
        # Query classification prompts
        self.classification_prompt = self._build_classification_prompt()
        
    async def initialize(self) -> bool:
        """Initialize the query router"""
        try:
            # Initialize LLM client
            if not await self.llm_client.initialize():
                logger.error("Failed to initialize LLM client")
                return False
            
            self.is_initialized = True
            logger.info("Query router initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize query router: {e}")
            return False
    
    def _build_classification_prompt(self) -> str:
        """Build the query classification prompt"""
        return """
You are an intelligent query router for a multi-database RAG system. Analyze the given query and determine the best approach for retrieval.

Available data sources:
- Vector Store: Contains embeddings of customer records, product information, transaction details, sales data, and business documents
- Graph Store: Contains relationships between databases, schemas, tables, and foreign key relationships

Query types:
1. VECTOR: Best for semantic similarity search, finding similar records, content-based queries
2. GRAPH: Best for relationship traversal, structural queries, schema exploration
3. HYBRID: Best for complex queries requiring both semantic search and relationship analysis

Consider these factors:
- Does the query ask about relationships, connections, or structure? → GRAPH
- Does the query ask about content, similarity, or specific data values? → VECTOR  
- Does the query require both content analysis and relationship understanding? → HYBRID

Query: "{query}"

Respond with valid JSON only:
{{
    "query_type": "vector|graph|hybrid",
    "confidence": 0.95,
    "reasoning": "Brief explanation of your choice",
    "vector_search_terms": ["term1", "term2"],
    "graph_patterns": ["pattern1", "pattern2"],
    "suggested_filters": {{"database": "WideWorldImporters", "schema": "Sales"}},
    "complexity_score": 0.7
}}
"""
    
    async def classify_query(self, query: str) -> QueryClassification:
        """Classify a query using LLM"""
        try:
            if not self.is_initialized:
                raise RuntimeError("Router not initialized")
            
            # Check cache first
            cache_key = hash(query.lower().strip())
            if cache_key in self.pattern_cache:
                return self.pattern_cache[cache_key]
            
            # Prepare prompt
            prompt = self.classification_prompt.format(query=query)
            
            # Get LLM response
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.1,  # Low temperature for consistent routing
                top_p=0.9
            )
            
            # Parse JSON response
            classification = self._parse_classification_response(response, query)
            
            # Cache the result
            self.pattern_cache[cache_key] = classification
            
            logger.info(f"Query classified as {classification.query_type.value} (confidence: {classification.confidence:.2f})")
            return classification
            
        except Exception as e:
            logger.error(f"Failed to classify query: {e}")
            # Return default classification
            return QueryClassification(
                query_type=QueryType.VECTOR,
                confidence=0.5,
                reasoning="Failed to classify, defaulting to vector search",
                vector_search_terms=self._extract_keywords(query),
                graph_patterns=[],
                suggested_filters={},
                complexity_score=0.5
            )
    
    def _parse_classification_response(self, response: str, original_query: str) -> QueryClassification:
        """Parse LLM response into QueryClassification"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            # Parse query type
            query_type_str = data.get("query_type", "vector").lower()
            if query_type_str == "vector":
                query_type = QueryType.VECTOR
            elif query_type_str == "graph":
                query_type = QueryType.GRAPH
            elif query_type_str == "hybrid":
                query_type = QueryType.HYBRID
            else:
                query_type = QueryType.VECTOR
            
            # Extract other fields
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "No reasoning provided")
            vector_search_terms = data.get("vector_search_terms", [])
            graph_patterns = data.get("graph_patterns", [])
            suggested_filters = data.get("suggested_filters", {})
            complexity_score = float(data.get("complexity_score", 0.5))
            
            # Validate and clean data
            if not isinstance(vector_search_terms, list):
                vector_search_terms = self._extract_keywords(original_query)
            
            if not isinstance(graph_patterns, list):
                graph_patterns = []
            
            if not isinstance(suggested_filters, dict):
                suggested_filters = {}
            
            return QueryClassification(
                query_type=query_type,
                confidence=max(0.0, min(1.0, confidence)),
                reasoning=reasoning,
                vector_search_terms=vector_search_terms,
                graph_patterns=graph_patterns,
                suggested_filters=suggested_filters,
                complexity_score=max(0.0, min(1.0, complexity_score))
            )
            
        except Exception as e:
            logger.error(f"Failed to parse classification response: {e}")
            logger.debug(f"Response was: {response}")
            
            # Return fallback classification
            return QueryClassification(
                query_type=QueryType.VECTOR,
                confidence=0.5,
                reasoning=f"Failed to parse LLM response: {e}",
                vector_search_terms=self._extract_keywords(original_query),
                graph_patterns=[],
                suggested_filters={},
                complexity_score=0.5
            )
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query as fallback"""
        # Simple keyword extraction
        stop_words = {
            "what", "how", "where", "when", "why", "who", "which", "the", "a", "an",
            "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "from", "about", "into", "through", "during", "before", "after", "above",
            "below", "between", "among", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "can", "show", "find", "get"
        }
        
        # Extract words, convert to lowercase, remove stop words
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:10]  # Limit to 10 keywords
    
    def analyze_query_patterns(self, query: str) -> Dict[str, Any]:
        """Analyze query patterns to help with routing"""
        patterns = {
            "has_relationship_keywords": False,
            "has_structural_keywords": False,
            "has_content_keywords": False,
            "has_similarity_keywords": False,
            "has_data_keywords": False,
            "complexity_indicators": []
        }
        
        query_lower = query.lower()
        
        # Relationship keywords
        relationship_keywords = [
            "relationship", "related", "connection", "connected", "link", "linked",
            "references", "depends", "foreign key", "primary key", "join", "joins"
        ]
        patterns["has_relationship_keywords"] = any(kw in query_lower for kw in relationship_keywords)
        
        # Structural keywords
        structural_keywords = [
            "schema", "table", "column", "database", "structure", "model", "entity",
            "hierarchy", "parent", "child", "contains", "belongs to"
        ]
        patterns["has_structural_keywords"] = any(kw in query_lower for kw in structural_keywords)
        
        # Content keywords
        content_keywords = [
            "customer", "product", "order", "sale", "purchase", "transaction", "data",
            "record", "information", "details", "description", "content"
        ]
        patterns["has_content_keywords"] = any(kw in query_lower for kw in content_keywords)
        
        # Similarity keywords
        similarity_keywords = [
            "similar", "like", "similar to", "comparable", "matching", "resembling",
            "find", "search", "show", "list", "get"
        ]
        patterns["has_similarity_keywords"] = any(kw in query_lower for kw in similarity_keywords)
        
        # Data keywords
        data_keywords = [
            "value", "amount", "price", "quantity", "total", "sum", "average",
            "count", "number", "date", "time", "name", "address"
        ]
        patterns["has_data_keywords"] = any(kw in query_lower for kw in data_keywords)
        
        # Complexity indicators
        if "and" in query_lower:
            patterns["complexity_indicators"].append("multiple_conditions")
        if "or" in query_lower:
            patterns["complexity_indicators"].append("alternative_conditions")
        if any(word in query_lower for word in ["path", "chain", "sequence", "flow"]):
            patterns["complexity_indicators"].append("path_analysis")
        if len(query.split()) > 15:
            patterns["complexity_indicators"].append("long_query")
        
        return patterns
    
    def get_routing_suggestion(self, query: str) -> QueryType:
        """Get routing suggestion based on pattern analysis"""
        patterns = self.analyze_query_patterns(query)
        
        # Score each query type
        vector_score = 0
        graph_score = 0
        hybrid_score = 0
        
        # Vector search indicators
        if patterns["has_content_keywords"]:
            vector_score += 2
        if patterns["has_similarity_keywords"]:
            vector_score += 2
        if patterns["has_data_keywords"]:
            vector_score += 1
        
        # Graph search indicators
        if patterns["has_relationship_keywords"]:
            graph_score += 3
        if patterns["has_structural_keywords"]:
            graph_score += 2
        
        # Hybrid indicators
        if len(patterns["complexity_indicators"]) > 1:
            hybrid_score += 2
        if patterns["has_content_keywords"] and patterns["has_relationship_keywords"]:
            hybrid_score += 3
        
        # Determine best approach
        if hybrid_score > max(vector_score, graph_score):
            return QueryType.HYBRID
        elif graph_score > vector_score:
            return QueryType.GRAPH
        else:
            return QueryType.VECTOR
    
    async def route_query(self, query: str) -> QueryClassification:
        """Route a query (main entry point)"""
        try:
            # Use LLM classification if available
            if self.is_initialized:
                return await self.classify_query(query)
            else:
                # Fall back to pattern-based routing
                suggested_type = self.get_routing_suggestion(query)
                
                return QueryClassification(
                    query_type=suggested_type,
                    confidence=0.7,
                    reasoning="Using pattern-based routing (LLM not available)",
                    vector_search_terms=self._extract_keywords(query),
                    graph_patterns=[],
                    suggested_filters={},
                    complexity_score=0.5
                )
                
        except Exception as e:
            logger.error(f"Failed to route query: {e}")
            # Return safe default
            return QueryClassification(
                query_type=QueryType.VECTOR,
                confidence=0.5,
                reasoning=f"Routing failed: {e}",
                vector_search_terms=self._extract_keywords(query),
                graph_patterns=[],
                suggested_filters={},
                complexity_score=0.5
            )
    
    def clear_cache(self):
        """Clear the pattern cache"""
        self.pattern_cache.clear()
        logger.info("Query router cache cleared")
    
    async def get_router_stats(self) -> Dict[str, Any]:
        """Get router statistics"""
        try:
            # Count cached patterns by type
            type_counts = {}
            for classification in self.pattern_cache.values():
                query_type = classification.query_type.value
                type_counts[query_type] = type_counts.get(query_type, 0) + 1
            
            return {
                "is_initialized": self.is_initialized,
                "cached_patterns": len(self.pattern_cache),
                "type_distribution": type_counts,
                "llm_backend": self.llm_client.current_backend.value if self.llm_client.current_backend else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get router stats: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Shutdown the router"""
        try:
            if self.llm_client:
                await self.llm_client.shutdown()
            
            self.is_initialized = False
            logger.info("Query router shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during router shutdown: {e}")


async def main():
    """Example usage of intelligent router"""
    router = IntelligentRouter()
    
    try:
        # Initialize router
        if not await router.initialize():
            logger.error("Failed to initialize router")
            return
        
        # Test queries
        test_queries = [
            "Show me customers with high purchase amounts",
            "What are the relationships between suppliers and products?",
            "Find sales data for customers in the electronics category",
            "How are the database schemas structured?",
            "Analyze customer purchasing patterns across product categories"
        ]
        
        for query in test_queries:
            print(f"\\nQuery: {query}")
            classification = await router.route_query(query)
            print(f"Routing: {classification.query_type.value} (confidence: {classification.confidence:.2f})")
            print(f"Reasoning: {classification.reasoning}")
            print(f"Vector terms: {classification.vector_search_terms}")
            print(f"Graph patterns: {classification.graph_patterns}")
        
        # Get stats
        stats = await router.get_router_stats()
        print(f"\\nRouter stats: {stats}")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        
    finally:
        await router.shutdown()


if __name__ == "__main__":
    asyncio.run(main())