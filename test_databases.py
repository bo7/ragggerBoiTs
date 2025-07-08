#!/usr/bin/env python3
"""
Test script for Milvus (vector) and Neo4j (graph) database connections
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from rag_system.storage import MilvusVectorStore, VectorStoreManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_milvus_connection():
    """Test Milvus vector database connection"""
    print("🗄️ Testing Milvus Vector Database Connection...")
    
    try:
        # Initialize Milvus client
        milvus = MilvusVectorStore(
            host="localhost",
            port=19530
        )
        
        # Connect to Milvus
        success = await milvus.initialize()
        if not success:
            print("❌ Failed to initialize Milvus connection")
            return False
        
        print("✅ Connected to Milvus successfully!")
        
        # Health check
        health = await milvus.health_check()
        print(f"🔍 Milvus health: {health}")
        
        # List existing collections
        collections = await milvus.list_collections()
        print(f"📚 Existing collections: {collections}")
        
        # Create a test collection
        collection_name = "test_collection"
        print(f"\n📝 Creating test collection: {collection_name}")
        
        success = await milvus.create_collection(
            collection_name=collection_name,
            dimension=2048,  # Jina v4 dimension
            description="Test collection for RAG system"
        )
        
        if success:
            print("✅ Test collection created successfully!")
            
            # Test data insertion
            print("\n📊 Testing vector insertion...")
            import numpy as np
            
            # Create sample vectors (simulating Jina embeddings)
            sample_vectors = np.random.rand(3, 2048).tolist()
            sample_texts = [
                "This is the first test document",
                "This is the second test document", 
                "This is the third test document"
            ]
            sample_metadata = [
                {"type": "test", "id": 1},
                {"type": "test", "id": 2},
                {"type": "test", "id": 3}
            ]
            
            insert_success = await milvus.insert_vectors(
                collection_name=collection_name,
                vectors=sample_vectors,
                texts=sample_texts,
                metadata=sample_metadata,
                source_table="test_table"
            )
            
            if insert_success:
                print("✅ Vector insertion successful!")
                
                # Test similarity search
                print("\n🔍 Testing similarity search...")
                query_vector = sample_vectors[0]  # Use first vector as query
                
                results = await milvus.search_similar(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=2
                )
                
                print(f"🎯 Search results: {len(results)} found")
                for i, result in enumerate(results):
                    print(f"   {i+1}. Score: {result['score']:.4f}, Text: {result['text'][:50]}...")
                
                # Get collection stats
                stats = await milvus.get_collection_stats(collection_name)
                print(f"\n📈 Collection stats: {stats}")
                
            else:
                print("❌ Vector insertion failed")
                return False
        else:
            print("❌ Failed to create test collection")
            return False
        
        # Clean up test collection
        print(f"\n🧹 Cleaning up test collection...")
        await milvus.delete_collection(collection_name)
        
        # Shutdown
        await milvus.shutdown()
        print("✅ Milvus test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Milvus test failed: {e}")
        print(f"❌ Milvus test failed: {e}")
        return False


def test_neo4j_connection():
    """Test Neo4j graph database connection"""
    print("\n🕸️ Testing Neo4j Graph Database Connection...")
    
    try:
        from neo4j import GraphDatabase
        
        # Connection details
        uri = "bolt://localhost:7687"
        username = "neo4j"
        password = "password123"
        
        print(f"🔗 Connecting to Neo4j at {uri}")
        
        # Create driver
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Test connection with a simple query
        with driver.session() as session:
            # Test basic connectivity
            result = session.run("RETURN 'Hello, Neo4j!' as message")
            record = result.single()
            print(f"✅ Neo4j connection successful: {record['message']}")
            
            # Create test nodes
            print("\n📝 Creating test nodes...")
            session.run("""
                CREATE (doc1:Document {id: 'test1', title: 'Test Document 1', content: 'This is test content 1'})
                CREATE (doc2:Document {id: 'test2', title: 'Test Document 2', content: 'This is test content 2'})
                CREATE (concept1:Concept {name: 'Machine Learning', category: 'AI'})
                CREATE (concept2:Concept {name: 'Data Science', category: 'Analytics'})
            """)
            
            # Create relationships
            print("🔗 Creating test relationships...")
            session.run("""
                MATCH (doc1:Document {id: 'test1'}), (concept1:Concept {name: 'Machine Learning'})
                CREATE (doc1)-[:MENTIONS]->(concept1)
            """)
            
            session.run("""
                MATCH (doc2:Document {id: 'test2'}), (concept2:Concept {name: 'Data Science'})
                CREATE (doc2)-[:MENTIONS]->(concept2)
            """)
            
            session.run("""
                MATCH (concept1:Concept {name: 'Machine Learning'}), (concept2:Concept {name: 'Data Science'})
                CREATE (concept1)-[:RELATED_TO {strength: 0.8}]->(concept2)
            """)
            
            # Query the graph
            print("\n🔍 Testing graph queries...")
            result = session.run("""
                MATCH (d:Document)-[:MENTIONS]->(c:Concept)
                RETURN d.title as document, c.name as concept
            """)
            
            print("📊 Document-Concept relationships:")
            for record in result:
                print(f"   {record['document']} mentions {record['concept']}")
            
            # Test graph traversal
            result = session.run("""
                MATCH (c1:Concept)-[r:RELATED_TO]->(c2:Concept)
                RETURN c1.name as concept1, c2.name as concept2, r.strength as strength
            """)
            
            print("\n🌐 Concept relationships:")
            for record in result:
                print(f"   {record['concept1']} -> {record['concept2']} (strength: {record['strength']})")
            
            # Get database info
            result = session.run("CALL db.labels()")
            labels = [record["label"] for record in result]
            print(f"\n🏷️ Available labels: {labels}")
            
            result = session.run("CALL db.relationshipTypes()")
            rel_types = [record["relationshipType"] for record in result]
            print(f"🔗 Available relationship types: {rel_types}")
            
            # Clean up test data
            print("\n🧹 Cleaning up test data...")
            session.run("MATCH (n) WHERE n.id IN ['test1', 'test2'] OR n.name IN ['Machine Learning', 'Data Science'] DETACH DELETE n")
        
        driver.close()
        print("✅ Neo4j test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Neo4j test failed: {e}")
        print(f"❌ Neo4j test failed: {e}")
        return False


async def test_vector_store_manager():
    """Test the high-level vector store manager"""
    print("\n🎛️ Testing Vector Store Manager...")
    
    try:
        # Initialize Milvus
        milvus = MilvusVectorStore(host="localhost", port=19530)
        await milvus.initialize()
        
        # Initialize manager
        manager = VectorStoreManager(milvus)
        
        # Setup RAG collections
        print("📚 Setting up RAG collections...")
        success = await manager.setup_rag_collections()
        
        if success:
            print("✅ RAG collections setup successful!")
            
            # List collections
            collections = await milvus.list_collections()
            print(f"📋 Available collections: {collections}")
            
            # Test with sample SQL data
            print("\n📊 Testing SQL data ingestion simulation...")
            
            sample_sql_data = [
                {"id": 1, "CustomerName": "John Doe", "City": "New York", "Country": "USA"},
                {"id": 2, "CustomerName": "Jane Smith", "City": "London", "Country": "UK"},
                {"id": 3, "CustomerName": "Bob Johnson", "City": "Toronto", "Country": "Canada"}
            ]
            
            # Simulate embeddings (normally from Jina)
            import numpy as np
            sample_embeddings = np.random.rand(len(sample_sql_data), 2048).tolist()
            
            # Ingest data
            ingest_success = await manager.ingest_sql_data(
                table_data=sample_sql_data,
                embeddings=sample_embeddings,
                table_name="Customers",
                collection_name="documents"
            )
            
            if ingest_success:
                print("✅ SQL data ingestion successful!")
                
                # Test search
                query_vector = sample_embeddings[0]
                results = await milvus.search_similar(
                    collection_name="documents",
                    query_vector=query_vector,
                    limit=2
                )
                
                print(f"🔍 Search results: {len(results)} found")
                for result in results:
                    print(f"   Score: {result['score']:.4f}, Source: {result['source_table']}")
            else:
                print("❌ SQL data ingestion failed")
        else:
            print("❌ RAG collections setup failed")
        
        await milvus.shutdown()
        return success
        
    except Exception as e:
        logger.error(f"Vector store manager test failed: {e}")
        print(f"❌ Vector store manager test failed: {e}")
        return False


async def main():
    """Main test function"""
    print("🧪 RAG System - Database Integration Test")
    print("=" * 50)
    
    # Test Milvus
    milvus_success = await test_milvus_connection()
    
    # Test Neo4j
    neo4j_success = test_neo4j_connection()
    
    # Test Vector Store Manager
    manager_success = await test_vector_store_manager()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   Milvus Vector DB: {'✅ PASS' if milvus_success else '❌ FAIL'}")
    print(f"   Neo4j Graph DB:   {'✅ PASS' if neo4j_success else '❌ FAIL'}")
    print(f"   Vector Manager:   {'✅ PASS' if manager_success else '❌ FAIL'}")
    
    if all([milvus_success, neo4j_success, manager_success]):
        print("\n🎉 All database tests passed! Ready for data pipeline integration!")
        return True
    else:
        print("\n❌ Some tests failed. Please check the database connections.")
        return False


if __name__ == "__main__":
    asyncio.run(main())