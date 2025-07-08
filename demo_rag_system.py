#!/usr/bin/env python3
"""
RAG System Demo
===============

Simple demo script to test the complete RAG system with real queries.
This script loads a small amount of data and demonstrates the LLM query routing.

Usage:
    python demo_rag_system.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_system.data_pipeline.multi_database_pipeline import MultiDatabasePipeline, DatabaseConfig
from rag_system.query.rag_engine import RAGEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


async def demo_rag_system():
    """Demo the complete RAG system"""
    print("🚀 RAG System Demo")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = MultiDatabasePipeline()
    rag_engine = RAGEngine()
    
    try:
        # Step 1: Initialize system
        print("\\n🔧 Initializing system...")
        if not await pipeline.initialize_pipeline():
            print("❌ Failed to initialize pipeline")
            return
        
        if not await rag_engine.initialize():
            print("❌ Failed to initialize RAG engine")
            return
        
        print("✅ System initialized")
        
        # Step 2: Load sample data
        print("\\n📊 Loading sample data...")
        
        # Configure for small demo
        config = DatabaseConfig(
            name="WideWorldImportersDW",
            enabled=True,
            sample_limit=10,  # Very small sample
            schemas_to_include=["Dimension"],
            tables_to_include=["Dimension.Customer", "Dimension.City"]
        )
        
        pipeline.add_database_config(config)
        
        # Process database
        stats = await pipeline.process_database("WideWorldImportersDW")
        print(f"✅ Loaded {stats.successful_embeddings} embeddings, {stats.vector_insertions} vectors")
        
        # Step 3: Demo queries
        print("\\n🤖 Testing query routing...")
        
        demo_queries = [
            "Show me customer information",
            "What is the database structure?",
            "Find customer data and relationships"
        ]
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\\n--- Query {i} ---")
            print(f"Question: {query}")
            
            # Execute query
            response = await rag_engine.query(query)
            
            print(f"🤖 Route: {response.query_type.upper()}")
            print(f"🎯 Confidence: {response.confidence:.2f}")
            print(f"⏱️  Time: {response.execution_time:.2f}s")
            print(f"📊 Sources: {len(response.sources)}")
            print(f"💬 Answer: {response.answer[:200]}...")
        
        print("\\n🎉 Demo completed successfully!")
        print("\\n📝 To run the full test suite:")
        print("   python test_complete_rag_system.py")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logger.error(f"Demo error: {e}")
        
    finally:
        # Cleanup
        await rag_engine.shutdown()
        await pipeline.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(demo_rag_system())
    except KeyboardInterrupt:
        print("\\n⚠️  Demo interrupted")
    except Exception as e:
        print(f"❌ Demo execution failed: {e}")
        sys.exit(1)