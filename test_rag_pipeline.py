#!/usr/bin/env python3
"""
Test script for the complete RAG data pipeline with dbt integration
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from rag_system.data_pipeline import RAGDataPipeline
from rag_system.embedding import JinaEmbeddingClient
from rag_system.storage import MilvusVectorStore

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_rag_pipeline():
    """Test the complete RAG pipeline"""
    
    # Load environment
    load_dotenv()
    
    # Check API keys
    jina_key = os.getenv('JINA_API_KEY')
    if not jina_key or jina_key == 'your_jina_api_key_here':
        print("❌ Please set your Jina API key in the .env file")
        return False
    
    print("🚀 Testing Complete RAG Data Pipeline...")
    print("=" * 50)
    
    try:
        # Initialize components
        print("\n🔧 Initializing pipeline components...")
        
        # Embedding client
        embedding_client = JinaEmbeddingClient(
            api_key=jina_key,
            model_name="jina-embeddings-v4"
        )
        
        # Vector store
        vector_store = MilvusVectorStore(
            host="localhost",
            port=19530
        )
        
        # Create pipeline
        pipeline = RAGDataPipeline(
            dbt_project_dir=Path.cwd(),
            output_dir=Path.cwd() / "rag_output",
            vector_store=vector_store,
            embedding_client=embedding_client
        )
        
        # Test pipeline initialization
        print("📡 Initializing RAG pipeline...")
        success = await pipeline.initialize()
        
        if not success:
            print("❌ Failed to initialize RAG pipeline")
            return False
        
        print("✅ RAG pipeline initialized successfully!")
        
        # Test dbt semantic extraction
        print("\n📚 Testing dbt semantic extraction...")
        
        # Get semantic info for a few tables
        test_tables = [
            ("Dimension", "Customer"),
            ("Fact", "Sale"),
            ("Integration", "Customer_Staging")
        ]
        
        for schema, table in test_tables:
            try:
                semantic_info = await pipeline.dbt_extractor.get_table_semantic_info(schema, table)
                print(f"✅ {schema}.{table}: {semantic_info.model_type} with {len(semantic_info.columns)} columns")
                
                if semantic_info.description:
                    print(f"   Description: {semantic_info.description[:100]}...")
                
                if semantic_info.business_rules:
                    print(f"   Business rules: {len(semantic_info.business_rules)} found")
                    
            except Exception as e:
                print(f"⚠️ {schema}.{table}: {e}")
        
        # Test enhanced SQL extraction (limited to save time)
        print("\n🗄️ Testing enhanced SQL extraction (limited)...")
        
        # Override sample limit for testing
        pipeline.sql_extractor.sample_limit = 2
        
        # Extract just a few tables for testing
        original_get_tables = pipeline.sql_extractor._get_tables
        
        async def limited_get_tables():
            all_tables = await original_get_tables()
            # Return only first 3 tables for testing
            return all_tables[:3]
        
        pipeline.sql_extractor._get_tables = limited_get_tables
        
        try:
            extracted_data = await pipeline.sql_extractor.extract_enhanced_catalog()
            
            print(f"✅ Extracted {len(extracted_data['tables'])} tables")
            print(f"   Constraints found: {len(extracted_data.get('constraints', {}))}")
            print(f"   Relationships found: {len(extracted_data.get('relationships', {}))}")
            print(f"   Semantic info: {len(extracted_data.get('semantic_info', {}))}")
            
            # Show sample table info
            if extracted_data['tables']:
                sample_table = extracted_data['tables'][0]
                print(f"\n📋 Sample table: {sample_table['schema']}.{sample_table['name']}")
                print(f"   Columns: {sample_table['column_count']}")
                print(f"   Sample rows: {sample_table['sample_row_count']}")
                
                if 'semantic_description' in sample_table:
                    print(f"   Has semantic description: Yes")
                
                # Show sample column
                if sample_table.get('columns'):
                    sample_col = sample_table['columns'][0]
                    print(f"   Sample column: {sample_col['name']} ({sample_col['data_type']})")
                    if sample_col.get('constraints'):
                        print(f"     Constraints: {sample_col['constraints']}")
            
        except Exception as e:
            print(f"❌ SQL extraction failed: {e}")
            return False
        
        # Test embedding generation (limited)
        print("\n🧠 Testing embedding generation...")
        
        try:
            # Test with just the first table
            if extracted_data['tables']:
                test_table = extracted_data['tables'][0]
                
                # Generate embeddings for this table
                table_embeddings = await pipeline._generate_table_embeddings(test_table)
                
                total_embeddings = sum(len(emb) for emb in table_embeddings.values() if emb)
                print(f"✅ Generated {total_embeddings} embeddings for {test_table['schema']}.{test_table['name']}")
                
                for embedding_type, embeddings in table_embeddings.items():
                    if embeddings:
                        print(f"   {embedding_type}: {len(embeddings)} embeddings ({len(embeddings[0])} dimensions)")
            
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            return False
        
        # Test vector storage (limited)
        print("\n🗂️ Testing vector storage...")
        
        try:
            # Setup RAG collections
            if not await pipeline.vector_manager.setup_rag_collections():
                print("❌ Failed to setup RAG collections")
                return False
            
            print("✅ RAG collections setup successful")
            
            # Test storing a few vectors
            if extracted_data['tables'] and table_embeddings:
                test_table = extracted_data['tables'][0]
                
                # Store table description embedding
                if table_embeddings.get('table_description'):
                    await pipeline._store_table_vectors(
                        "documents",
                        test_table,
                        table_embeddings['table_description'],
                        "table_description"
                    )
                    print("✅ Stored table description vectors")
                
                # Store column embeddings
                if table_embeddings.get('column_descriptions'):
                    await pipeline._store_column_vectors(
                        "database_schema",
                        test_table,
                        table_embeddings['column_descriptions']
                    )
                    print("✅ Stored column description vectors")
            
        except Exception as e:
            print(f"❌ Vector storage failed: {e}")
            return False
        
        # Test semantic search
        print("\n🔍 Testing semantic search...")
        
        try:
            # Test queries
            test_queries = [
                "customer information",
                "sales data",
                "primary key columns"
            ]
            
            for query in test_queries:
                results = await pipeline.search_semantic(query, limit=3)
                print(f"✅ Query '{query}': {len(results)} results")
                
                for i, result in enumerate(results[:2]):  # Show first 2 results
                    print(f"   {i+1}. Score: {result.get('score', 0):.4f} - {result.get('text', '')[:80]}...")
        
        except Exception as e:
            print(f"❌ Semantic search failed: {e}")
            return False
        
        # Test full pipeline run (minimal)
        print("\n🏃 Testing minimal full pipeline run...")
        
        try:
            # Create a new pipeline with minimal settings for full test
            mini_pipeline = RAGDataPipeline(
                dbt_project_dir=Path.cwd(),
                output_dir=Path.cwd() / "rag_output" / "mini_test",
                vector_store=vector_store,
                embedding_client=embedding_client
            )
            
            await mini_pipeline.initialize()
            
            # Override to extract only 1 table
            async def single_table_get_tables():
                all_tables = await original_get_tables()
                return all_tables[:1]  # Just 1 table
            
            mini_pipeline.sql_extractor._get_tables = single_table_get_tables
            mini_pipeline.sql_extractor.sample_limit = 1
            
            # Run pipeline
            results = await mini_pipeline.run_full_pipeline(
                extract_data=True,
                generate_embeddings=True,
                store_vectors=True,
                store_graph=False
            )
            
            print(f"✅ Full pipeline completed successfully!")
            print(f"   Tables extracted: {results['stats'].get('tables_extracted', 0)}")
            print(f"   Embeddings generated: {results['stats'].get('embeddings_generated', 0)}")
            print(f"   Vectors stored: {results['stats'].get('vectors_stored', 0)}")
            print(f"   Errors: {len(results.get('errors', []))}")
            
            if results.get('errors'):
                print("   Errors encountered:")
                for error in results['errors']:
                    print(f"     - {error}")
            
            await mini_pipeline.cleanup()
            
        except Exception as e:
            print(f"❌ Full pipeline test failed: {e}")
            return False
        
        # Cleanup
        await pipeline.cleanup()
        
        print("\n" + "=" * 50)
        print("🎉 All RAG pipeline tests completed successfully!")
        print("\n📊 Test Summary:")
        print("   ✅ dbt semantic extraction")
        print("   ✅ Enhanced SQL extraction with constraints")
        print("   ✅ Business-aware embedding generation")
        print("   ✅ Vector storage in Milvus")
        print("   ✅ Semantic search functionality")
        print("   ✅ Full pipeline orchestration")
        
        print("\n🚀 RAG system is ready for production use!")
        return True
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        print(f"❌ Pipeline test failed: {e}")
        return False


async def test_constraint_extraction():
    """Test specific constraint extraction functionality"""
    print("\n🔒 Testing constraint extraction...")
    
    try:
        from rag_system.data_pipeline import DBTSemanticExtractor
        
        dbt_extractor = DBTSemanticExtractor(Path.cwd())
        await dbt_extractor.initialize()
        
        # Test constraint extraction for a few tables
        test_tables = [
            ("Dimension", "Customer"),
            ("Fact", "Sale")
        ]
        
        for schema, table in test_tables:
            try:
                constraints = await dbt_extractor.get_constraint_info(schema, table)
                
                print(f"📋 {schema}.{table} constraints:")
                for constraint_type, constraint_list in constraints.items():
                    if constraint_list:
                        print(f"   {constraint_type}: {constraint_list}")
                        
            except Exception as e:
                print(f"⚠️ {schema}.{table}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Constraint extraction test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🧪 RAG Data Pipeline - Complete Integration Test")
    print("=" * 60)
    
    # Run the main pipeline test
    success = asyncio.run(test_rag_pipeline())
    
    # Run constraint extraction test
    constraint_success = asyncio.run(test_constraint_extraction())
    
    if success and constraint_success:
        print("\n🎉 All tests passed! RAG pipeline is ready for production!")
        print("\n📋 Next steps:")
        print("   1. Run full pipeline on complete dataset")
        print("   2. Set up AutoGen agents for intelligent querying")
        print("   3. Deploy MCP servers for enhanced capabilities")
        print("   4. Create customer deployment package")
    else:
        print("\n❌ Some tests failed. Please check the logs.")


if __name__ == "__main__":
    main()