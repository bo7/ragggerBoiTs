#!/usr/bin/env python3
"""
Complete End-to-End RAG System Test
===================================

Comprehensive test that demonstrates the full RAG pipeline with intelligent 
query routing. This test:

1. Clears existing vector/graph data
2. Loads both WideWorldImporters and WideWorldImportersDW databases
3. Tests LLM-powered query routing (vector vs graph vs hybrid)
4. Demonstrates interactive querying with explanations
5. Shows performance metrics and system health

Run this test to validate the complete RAG system functionality.
"""

import asyncio
import logging
import json
import sys
from pathlib import Path
from datetime import datetime
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_system.data_pipeline.multi_database_pipeline import MultiDatabasePipeline, DatabaseConfig
from rag_system.query.rag_engine import RAGEngine
from rag_system.query.intelligent_router import IntelligentRouter
from rag_system.storage.vector_store import MilvusVectorStore
from rag_system.storage.graph_store import Neo4jGraphStore
from rag_system.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress some verbose logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


class CompleteRAGTester:
    """Complete RAG system tester"""
    
    def __init__(self):
        self.pipeline = MultiDatabasePipeline()
        self.rag_engine = None  # Will be initialized after pipeline
        self.test_results = {}
        self.start_time = None
        
    async def run_complete_test(self) -> bool:
        """Run the complete RAG system test"""
        self.start_time = datetime.now()
        
        try:
            print("🚀 Complete RAG System Test")
            print("=" * 50)
            
            # Phase 1: Environment Reset
            success = await self._phase_1_environment_reset()
            if not success:
                return False
            
            # Phase 2: Data Loading
            success = await self._phase_2_data_loading()
            if not success:
                return False
            
            # Phase 3: Query Routing Tests
            success = await self._phase_3_query_routing_tests()
            if not success:
                return False
            
            # Phase 4: Interactive Demo
            success = await self._phase_4_interactive_demo()
            if not success:
                return False
            
            # Phase 5: Performance Metrics
            await self._phase_5_performance_metrics()
            
            # Final Summary
            await self._print_final_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"Complete test failed: {e}")
            return False
        finally:
            await self._cleanup()
    
    async def _phase_1_environment_reset(self) -> bool:
        """Phase 1: Clean environment and initialize"""
        print("\\n✅ Phase 1: Environment Reset")
        print("-" * 30)
        
        try:
            # Initialize pipeline
            print("🔧 Initializing pipeline...")
            if not await self.pipeline.initialize_pipeline():
                print("❌ Failed to initialize pipeline")
                return False
            
            # Initialize RAG engine with the same vector and graph stores
            self.rag_engine = RAGEngine(
                vector_store=self.pipeline.vector_store,
                graph_store=self.pipeline.graph_store
            )
            
            # Clear vector database
            print("🗑️  Clearing vector database...")
            vector_collections = await self.pipeline.vector_store.list_collections()
            for collection in vector_collections:
                await self.pipeline.vector_store.delete_collection(collection)
            
            # Clear graph database
            print("🗑️  Clearing graph database...")
            await self.pipeline.graph_store.clear_database()
            
            # Recreate collections and indexes
            print("🏗️  Setting up fresh collections...")
            await self.pipeline.vector_manager.setup_rag_collections()
            await self.pipeline.graph_store.create_indexes()
            
            # Initialize RAG engine
            print("🤖 Initializing RAG engine...")
            if not await self.rag_engine.initialize():
                print("❌ Failed to initialize RAG engine")
                return False
            
            print("✅ Environment reset completed")
            self.test_results["phase_1"] = {"success": True, "message": "Environment reset successful"}
            return True
            
        except Exception as e:
            print(f"❌ Environment reset failed: {e}")
            self.test_results["phase_1"] = {"success": False, "error": str(e)}
            return False
    
    async def _phase_2_data_loading(self) -> bool:
        """Phase 2: Load data from both databases"""
        print("\\n✅ Phase 2: Data Loading")
        print("-" * 30)
        
        try:
            # Configure databases for testing (smaller samples)
            oltp_config = DatabaseConfig(
                name="WideWorldImporters",
                enabled=True,
                priority=1,
                sample_limit=20,  # Small sample for testing
                schemas_to_include=["Application", "Sales"],
                tables_to_include=["Application.Cities", "Sales.Customers"]
            )
            
            dw_config = DatabaseConfig(
                name="WideWorldImportersDW",
                enabled=True,
                priority=2,
                sample_limit=20,  # Small sample for testing
                schemas_to_include=["Dimension"],
                tables_to_include=["Dimension.City", "Dimension.Customer"]
            )
            
            self.pipeline.add_database_config(oltp_config)
            self.pipeline.add_database_config(dw_config)
            
            # Process databases
            print("📊 Processing WideWorldImporters (OLTP)...")
            oltp_stats = await self.pipeline.process_database("WideWorldImporters")
            
            print("📊 Processing WideWorldImportersDW (DW)...")
            dw_stats = await self.pipeline.process_database("WideWorldImportersDW")
            
            # Verify data loading
            vector_health = await self.pipeline.vector_store.health_check()
            graph_health = await self.pipeline.graph_store.health_check()
            
            print(f"✅ OLTP: {oltp_stats.successful_embeddings} embeddings, {oltp_stats.vector_insertions} vectors")
            print(f"✅ DW: {dw_stats.successful_embeddings} embeddings, {dw_stats.vector_insertions} vectors")
            print(f"✅ Vector store: {vector_health.get('collections', [])} collections")
            print(f"✅ Graph store: {graph_health.get('stats', {}).get('node_count', 0)} nodes")
            
            self.test_results["phase_2"] = {
                "success": True,
                "oltp_stats": {
                    "embeddings": oltp_stats.successful_embeddings,
                    "vectors": oltp_stats.vector_insertions
                },
                "dw_stats": {
                    "embeddings": dw_stats.successful_embeddings,
                    "vectors": dw_stats.vector_insertions
                },
                "vector_collections": len(vector_health.get('collections', [])),
                "graph_nodes": graph_health.get('stats', {}).get('node_count', 0)
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            self.test_results["phase_2"] = {"success": False, "error": str(e)}
            return False
    
    async def _phase_3_query_routing_tests(self) -> bool:
        """Phase 3: Test query routing with different query types"""
        print("\\n✅ Phase 3: Query Routing Tests")
        print("-" * 30)
        
        try:
            # Test queries for different routing types
            test_queries = [
                {
                    "query": "Show me information about customers",
                    "expected_type": "vector",
                    "description": "Content-based query"
                },
                {
                    "query": "What is the database schema structure?",
                    "expected_type": "graph",
                    "description": "Structure/relationship query"
                },
                {
                    "query": "Find customer data and show how databases relate",
                    "expected_type": "hybrid",
                    "description": "Complex query requiring both approaches"
                },
                {
                    "query": "List cities in the database",
                    "expected_type": "vector",
                    "description": "Data retrieval query"
                },
                {
                    "query": "How are tables connected in the system?",
                    "expected_type": "graph",
                    "description": "Relationship exploration query"
                }
            ]
            
            routing_results = []
            
            for i, test_case in enumerate(test_queries, 1):
                print(f"\\n🔍 Test {i}: {test_case['description']}")
                print(f"Query: '{test_case['query']}'")
                
                # Get routing decision
                classification = await self.rag_engine.router.route_query(test_case['query'])
                
                print(f"🤖 LLM Decision: {classification.query_type.value} (confidence: {classification.confidence:.2f})")
                print(f"💭 Reasoning: {classification.reasoning}")
                
                # Execute query
                response = await self.rag_engine.query(test_case['query'])
                
                print(f"📊 Results: {len(response.sources)} sources found")
                print(f"⏱️  Execution time: {response.execution_time:.2f}s")
                print(f"📝 Answer preview: {response.answer[:100]}...")
                
                routing_results.append({
                    "query": test_case['query'],
                    "expected_type": test_case['expected_type'],
                    "actual_type": classification.query_type.value,
                    "confidence": classification.confidence,
                    "execution_time": response.execution_time,
                    "sources_found": len(response.sources)
                })
            
            # Calculate routing accuracy
            correct_routes = sum(1 for r in routing_results if r['expected_type'] == r['actual_type'])
            routing_accuracy = correct_routes / len(routing_results) * 100
            
            print(f"\\n📈 Routing Accuracy: {routing_accuracy:.1f}% ({correct_routes}/{len(routing_results)})")
            
            self.test_results["phase_3"] = {
                "success": True,
                "routing_accuracy": routing_accuracy,
                "test_results": routing_results
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Query routing tests failed: {e}")
            self.test_results["phase_3"] = {"success": False, "error": str(e)}
            return False
    
    async def _phase_4_interactive_demo(self) -> bool:
        """Phase 4: Interactive demonstration"""
        print("\\n✅ Phase 4: Interactive Demo")
        print("-" * 30)
        
        try:
            # Demo queries that show different aspects
            demo_queries = [
                "Show me customer information from the databases",
                "What databases and schemas are available?",
                "Find cities and their relationships across systems"
            ]
            
            demo_results = []
            
            for i, query in enumerate(demo_queries, 1):
                print(f"\\n🎯 Demo {i}: {query}")
                print("=" * 60)
                
                # Execute query
                response = await self.rag_engine.query(query)
                
                print(f"🤖 Query Type: {response.query_type.upper()}")
                print(f"🎯 Confidence: {response.confidence:.2f}")
                print(f"⏱️  Execution Time: {response.execution_time:.2f}s")
                print(f"📊 Sources: {len(response.sources)}")
                
                print("\\n💬 Answer:")
                print("-" * 20)
                print(response.answer)
                
                if response.sources:
                    print("\\n📚 Sources:")
                    for j, source in enumerate(response.sources[:3], 1):
                        if source.get('type') == 'vector':
                            print(f"  {j}. Vector: {source.get('database', 'Unknown')}.{source.get('schema', 'Unknown')}.{source.get('table', 'Unknown')} (score: {source.get('score', 0):.3f})")
                        elif source.get('type') == 'graph':
                            print(f"  {j}. Graph: Relationship data")
                
                demo_results.append({
                    "query": query,
                    "query_type": response.query_type,
                    "confidence": response.confidence,
                    "execution_time": response.execution_time,
                    "sources_count": len(response.sources)
                })
            
            self.test_results["phase_4"] = {
                "success": True,
                "demo_results": demo_results
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Interactive demo failed: {e}")
            self.test_results["phase_4"] = {"success": False, "error": str(e)}
            return False
    
    async def _phase_5_performance_metrics(self):
        """Phase 5: Collect performance metrics"""
        print("\\n✅ Phase 5: Performance Metrics")
        print("-" * 30)
        
        try:
            # Get system health
            engine_stats = await self.rag_engine.get_engine_stats()
            
            # Calculate overall metrics
            total_time = (datetime.now() - self.start_time).total_seconds()
            
            # Get query performance from previous phases
            routing_tests = self.test_results.get("phase_3", {}).get("test_results", [])
            demo_tests = self.test_results.get("phase_4", {}).get("demo_results", [])
            
            all_queries = routing_tests + demo_tests
            
            if all_queries:
                avg_query_time = sum(q.get("execution_time", 0) for q in all_queries) / len(all_queries)
                avg_confidence = sum(q.get("confidence", 0) for q in all_queries) / len(all_queries)
                total_sources = sum(q.get("sources_count", 0) for q in all_queries)
            else:
                avg_query_time = 0
                avg_confidence = 0
                total_sources = 0
            
            print(f"📊 Total Test Duration: {total_time:.2f}s")
            print(f"⚡ Average Query Time: {avg_query_time:.2f}s")
            print(f"🎯 Average Confidence: {avg_confidence:.2f}")
            print(f"📚 Total Sources Retrieved: {total_sources}")
            
            # Component health
            print("\\n🏥 Component Health:")
            components = engine_stats.get("components", {})
            for component, health in components.items():
                if isinstance(health, dict):
                    healthy = health.get("healthy", False)
                    status = "✅ Healthy" if healthy else "❌ Unhealthy"
                    print(f"  {component}: {status}")
            
            self.test_results["phase_5"] = {
                "total_time": total_time,
                "avg_query_time": avg_query_time,
                "avg_confidence": avg_confidence,
                "total_sources": total_sources,
                "component_health": components
            }
            
        except Exception as e:
            print(f"❌ Performance metrics failed: {e}")
            self.test_results["phase_5"] = {"success": False, "error": str(e)}
    
    async def _print_final_summary(self):
        """Print final test summary"""
        print("\\n🎉 Complete RAG System: TEST SUMMARY")
        print("=" * 50)
        
        # Count successful phases
        successful_phases = sum(1 for phase in self.test_results.values() if phase.get("success", False))
        total_phases = len([k for k in self.test_results.keys() if k.startswith("phase_")])
        
        print(f"✅ Phases Passed: {successful_phases}/{total_phases}")
        
        # Phase-by-phase summary
        for phase_name, results in self.test_results.items():
            if phase_name.startswith("phase_"):
                phase_num = phase_name.split("_")[1]
                status = "✅ PASSED" if results.get("success", False) else "❌ FAILED"
                print(f"  Phase {phase_num}: {status}")
        
        # Key metrics
        if self.test_results.get("phase_2", {}).get("success"):
            phase_2 = self.test_results["phase_2"]
            total_embeddings = phase_2.get("oltp_stats", {}).get("embeddings", 0) + phase_2.get("dw_stats", {}).get("embeddings", 0)
            total_vectors = phase_2.get("oltp_stats", {}).get("vectors", 0) + phase_2.get("dw_stats", {}).get("vectors", 0)
            print(f"📊 Data Loaded: {total_embeddings} embeddings → {total_vectors} vectors")
        
        if self.test_results.get("phase_3", {}).get("success"):
            routing_accuracy = self.test_results["phase_3"].get("routing_accuracy", 0)
            print(f"🤖 LLM Routing Accuracy: {routing_accuracy:.1f}%")
        
        if self.test_results.get("phase_5", {}).get("success"):
            avg_time = self.test_results["phase_5"].get("avg_query_time", 0)
            avg_confidence = self.test_results["phase_5"].get("avg_confidence", 0)
            print(f"⚡ Average Query Time: {avg_time:.2f}s")
            print(f"🎯 Average Confidence: {avg_confidence:.2f}")
        
        # Overall status
        if successful_phases == total_phases:
            print("\\n🎉 COMPLETE RAG SYSTEM: FULLY OPERATIONAL")
            print("✅ All components working correctly")
            print("✅ Multi-database integration successful")
            print("✅ LLM query routing functional")
            print("✅ Vector and graph search operational")
            print("✅ End-to-end pipeline validated")
        else:
            print("\\n❌ COMPLETE RAG SYSTEM: ISSUES DETECTED")
            print(f"⚠️  {total_phases - successful_phases} phases failed")
            print("🔧 Check logs for detailed error information")
    
    async def _cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'rag_engine'):
                await self.rag_engine.shutdown()
            if hasattr(self, 'pipeline'):
                await self.pipeline.shutdown()
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    async def export_test_results(self, filename: str = None) -> str:
        """Export test results to JSON file"""
        try:
            if filename is None:
                filename = f"complete_rag_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            
            print(f"\\n📄 Test results exported to: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to export test results: {e}")
            return None


async def main():
    """Main test execution"""
    print("🚀 Starting Complete RAG System Test")
    print("This will test the full pipeline with real data loading and LLM routing")
    print()
    
    # Check if Docker services are running
    print("🐳 Checking Docker services...")
    try:
        import docker
        client = docker.from_env()
        
        # Check for required containers
        required_containers = ['milvus-standalone', 'neo4j-graph']
        running_containers = [c.name for c in client.containers.list() if c.status == 'running']
        
        missing_containers = [name for name in required_containers if name not in running_containers]
        
        if missing_containers:
            print(f"❌ Missing containers: {missing_containers}")
            print("🔧 Please run: docker-compose up -d")
            return False
        else:
            print("✅ All required containers are running")
            
    except Exception as e:
        print(f"⚠️  Could not check Docker containers: {e}")
        print("🔧 Please ensure Milvus and Neo4j are running")
    
    # Run the test
    tester = CompleteRAGTester()
    
    try:
        success = await tester.run_complete_test()
        
        # Export results
        await tester.export_test_results()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())