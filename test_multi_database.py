#!/usr/bin/env python3
"""
Multi-Database RAG Pipeline Test
================================

Test script for the multi-database RAG pipeline functionality.
Tests database discovery, processing, and unified storage.
"""

import asyncio
import logging
import json
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_system.data_pipeline.multi_database_pipeline import MultiDatabasePipeline, DatabaseConfig
from rag_system.data_pipeline.database_discovery import DatabaseDiscovery
from rag_system.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_database_discovery():
    """Test database discovery functionality"""
    logger.info("Testing database discovery...")
    
    discovery = DatabaseDiscovery()
    
    # Test both databases
    databases = ['WideWorldImporters', 'WideWorldImportersDW']
    
    for db_name in databases:
        try:
            logger.info(f"Discovering schema for {db_name}...")
            schema = await discovery.discover_database_schema(db_name)
            
            if schema:
                logger.info(f"✓ {db_name}: {len(schema)} tables discovered")
                
                # Show some sample tables
                sample_tables = list(schema.keys())[:3]
                for table_name in sample_tables:
                    table_info = schema[table_name]
                    logger.info(f"  - {table_name}: {len(table_info.columns)} columns, "
                               f"{table_info.row_count} rows")
            else:
                logger.error(f"✗ {db_name}: No schema discovered")
                
        except Exception as e:
            logger.error(f"✗ {db_name}: Discovery failed - {e}")
    
    return True


async def test_pipeline_initialization():
    """Test pipeline initialization"""
    logger.info("Testing pipeline initialization...")
    
    pipeline = MultiDatabasePipeline()
    
    try:
        # Test initialization
        success = await pipeline.initialize_pipeline()
        
        if success:
            logger.info("✓ Pipeline initialized successfully")
            
            # Test component health
            vector_health = await pipeline.vector_store.health_check()
            logger.info(f"✓ Vector store health: {vector_health}")
            
            graph_health = await pipeline.graph_store.health_check()
            logger.info(f"✓ Graph store health: {graph_health}")
            
            embedding_health = await pipeline.embedding_client.health_check()
            logger.info(f"✓ Embedding client health: {embedding_health}")
            
            return True
        else:
            logger.error("✗ Pipeline initialization failed")
            return False
            
    except Exception as e:
        logger.error(f"✗ Pipeline initialization error: {e}")
        return False
    finally:
        await pipeline.shutdown()


async def test_single_database_processing():
    """Test processing a single database"""
    logger.info("Testing single database processing...")
    
    pipeline = MultiDatabasePipeline()
    
    try:
        # Initialize pipeline
        if not await pipeline.initialize_pipeline():
            logger.error("✗ Pipeline initialization failed")
            return False
        
        # Configure for limited processing
        config = DatabaseConfig(
            name="WideWorldImportersDW",
            enabled=True,
            priority=1,
            sample_limit=10,  # Small sample for testing
            schemas_to_include=["Dimension"],
            tables_to_include=["Dimension.City", "Dimension.Customer"]
        )
        
        pipeline.add_database_config(config)
        
        # Process database
        stats = await pipeline.process_database("WideWorldImportersDW")
        
        if stats.processed_tables > 0:
            logger.info(f"✓ Processed {stats.processed_tables} tables")
            logger.info(f"  - Records: {stats.total_records}")
            logger.info(f"  - Embeddings: {stats.successful_embeddings}")
            logger.info(f"  - Success rate: {stats.success_rate:.1f}%")
            logger.info(f"  - Duration: {stats.duration:.1f}s")
            
            if stats.errors:
                logger.warning(f"  - Errors: {len(stats.errors)}")
                for error in stats.errors[:3]:  # Show first 3 errors
                    logger.warning(f"    {error}")
            
            return True
        else:
            logger.error("✗ No tables processed")
            return False
            
    except Exception as e:
        logger.error(f"✗ Single database processing failed: {e}")
        return False
    finally:
        await pipeline.shutdown()


async def test_multi_database_processing():
    """Test full multi-database processing"""
    logger.info("Testing multi-database processing...")
    
    pipeline = MultiDatabasePipeline()
    
    try:
        # Initialize pipeline
        if not await pipeline.initialize_pipeline():
            logger.error("✗ Pipeline initialization failed")
            return False
        
        # Configure for limited processing
        oltp_config = DatabaseConfig(
            name="WideWorldImporters",
            enabled=True,
            priority=1,
            sample_limit=5,
            schemas_to_include=["Application", "Sales"],
            tables_to_include=["Application.Cities", "Sales.Customers"]
        )
        
        dw_config = DatabaseConfig(
            name="WideWorldImportersDW",
            enabled=True,
            priority=2,
            sample_limit=5,
            schemas_to_include=["Dimension"],
            tables_to_include=["Dimension.City", "Dimension.Customer"]
        )
        
        pipeline.add_database_config(oltp_config)
        pipeline.add_database_config(dw_config)
        
        # Process all databases
        all_stats = await pipeline.process_all_databases()
        
        # Generate summary
        summary = await pipeline.get_processing_summary(all_stats)
        
        logger.info(f"✓ Multi-database processing completed")
        logger.info(f"  - Databases: {summary['total_databases']}")
        logger.info(f"  - Tables: {summary['processed_tables']}/{summary['total_tables']}")
        logger.info(f"  - Records: {summary['total_records']:,}")
        logger.info(f"  - Embeddings: {summary['successful_embeddings']:,}")
        logger.info(f"  - Vector insertions: {summary['vector_insertions']:,}")
        logger.info(f"  - Graph insertions: {summary['graph_insertions']:,}")
        logger.info(f"  - Total errors: {summary['total_errors']}")
        logger.info(f"  - Processing time: {summary['processing_time']:.1f}s")
        
        # Export results
        results_file = await pipeline.export_processing_results(all_stats)
        if results_file:
            logger.info(f"✓ Results exported to: {results_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Multi-database processing failed: {e}")
        return False
    finally:
        await pipeline.shutdown()


async def test_vector_search():
    """Test vector search across multiple databases"""
    logger.info("Testing vector search across databases...")
    
    pipeline = MultiDatabasePipeline()
    
    try:
        # Initialize pipeline
        if not await pipeline.initialize_pipeline():
            logger.error("✗ Pipeline initialization failed")
            return False
        
        # Test search
        query_text = "customer information"
        
        # Generate query embedding
        query_embedding = await pipeline.embedding_client.embed_texts(
            [query_text],
            task_type="text-embedding-ada-002"
        )
        
        if not query_embedding:
            logger.error("✗ Failed to generate query embedding")
            return False
        
        # Search in vector store
        results = await pipeline.vector_store.search_similar(
            collection_name="documents",
            query_vector=query_embedding[0],
            limit=5
        )
        
        if results:
            logger.info(f"✓ Found {len(results)} similar documents")
            for i, result in enumerate(results):
                logger.info(f"  {i+1}. {result['source_database']}.{result['source_schema']}.{result['source_table']}")
                logger.info(f"     Score: {result['score']:.3f}")
                logger.info(f"     Text: {result['text'][:100]}...")
                
            return True
        else:
            logger.warning("✗ No search results found")
            return False
            
    except Exception as e:
        logger.error(f"✗ Vector search failed: {e}")
        return False
    finally:
        await pipeline.shutdown()


async def test_graph_relationships():
    """Test graph relationship queries"""
    logger.info("Testing graph relationship queries...")
    
    pipeline = MultiDatabasePipeline()
    
    try:
        # Initialize pipeline
        if not await pipeline.initialize_pipeline():
            logger.error("✗ Pipeline initialization failed")
            return False
        
        # Test graph queries
        query = """
        MATCH (db:Database)-[:CONTAINS]->(schema:Schema)-[:CONTAINS]->(table:Table)
        RETURN db.database, schema.schema, table.table, table.row_count
        ORDER BY db.database, schema.schema, table.table
        LIMIT 10
        """
        
        results = await pipeline.graph_store.execute_query(query)
        
        if results:
            logger.info(f"✓ Found {len(results)} database relationships")
            for result in results:
                logger.info(f"  - {result['db.database']}.{result['schema.schema']}.{result['table.table']} "
                           f"({result['table.row_count']} rows)")
            
            return True
        else:
            logger.warning("✗ No graph relationships found")
            return False
            
    except Exception as e:
        logger.error(f"✗ Graph relationship query failed: {e}")
        return False
    finally:
        await pipeline.shutdown()


async def run_all_tests():
    """Run all tests"""
    logger.info("Starting multi-database RAG pipeline tests...")
    
    tests = [
        ("Database Discovery", test_database_discovery),
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Single Database Processing", test_single_database_processing),
        ("Multi-Database Processing", test_multi_database_processing),
        ("Vector Search", test_vector_search),
        ("Graph Relationships", test_graph_relationships)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = await test_func()
            results[test_name] = success
            
            if success:
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
                
        except Exception as e:
            logger.error(f"✗ {test_name} ERROR: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    try:
        # Set up environment
        settings = get_settings()
        logger.info(f"Using database: {settings.db_server}")
        
        # Run tests
        success = asyncio.run(run_all_tests())
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("\\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)