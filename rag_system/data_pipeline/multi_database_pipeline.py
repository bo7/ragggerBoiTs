"""
Multi-Database RAG Pipeline
===========================

Orchestrates RAG pipeline processing across multiple databases with unified storage.
Supports both WideWorldImporters and WideWorldImportersDW databases.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import time
from datetime import datetime

from .database_discovery import DatabaseDiscovery
from .dbt_integration import DBTSemanticExtractor
from .sql_extractor import EnhancedSQLExtractor
from ..embedding.jina_client import JinaEmbeddingClient
from ..storage.vector_store import MilvusVectorStore, VectorStoreManager
from ..storage.graph_store import Neo4jGraphStore
from ..config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Configuration for a database in the multi-database pipeline"""
    name: str
    enabled: bool = True
    priority: int = 1
    sample_limit: int = 100
    schemas_to_include: List[str] = None
    schemas_to_exclude: List[str] = None
    tables_to_include: List[str] = None
    tables_to_exclude: List[str] = None
    
    def __post_init__(self):
        if self.schemas_to_include is None:
            self.schemas_to_include = []
        if self.schemas_to_exclude is None:
            self.schemas_to_exclude = []
        if self.tables_to_include is None:
            self.tables_to_include = []
        if self.tables_to_exclude is None:
            self.tables_to_exclude = []


@dataclass
class ProcessingStats:
    """Statistics from pipeline processing"""
    database_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_tables: int = 0
    processed_tables: int = 0
    total_records: int = 0
    successful_embeddings: int = 0
    failed_embeddings: int = 0
    vector_insertions: int = 0
    graph_insertions: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    @property
    def duration(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_records > 0:
            return (self.successful_embeddings / self.total_records) * 100
        return 0.0


class MultiDatabasePipeline:
    """Multi-database RAG pipeline orchestrator"""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize components
        self.discovery = DatabaseDiscovery()
        self.dbt_integration = DBTSemanticExtractor()
        self.sql_extractor = EnhancedSQLExtractor()
        self.embedding_client = JinaEmbeddingClient()
        
        # Storage components
        self.vector_store = MilvusVectorStore()
        self.vector_manager = VectorStoreManager(self.vector_store)
        self.graph_store = Neo4jGraphStore()
        
        # Configuration
        self.database_configs = {}
        self.processing_stats = {}
        
        # Initialize default database configurations
        self._setup_default_configs()
    
    def _setup_default_configs(self):
        """Setup default database configurations"""
        self.database_configs = {
            "WideWorldImporters": DatabaseConfig(
                name="WideWorldImporters",
                enabled=True,
                priority=1,
                sample_limit=100,
                schemas_to_include=["Application", "Sales", "Purchasing", "Warehouse"],
                schemas_to_exclude=["sys", "INFORMATION_SCHEMA"]
            ),
            "WideWorldImportersDW": DatabaseConfig(
                name="WideWorldImportersDW",
                enabled=True,
                priority=2,
                sample_limit=100,
                schemas_to_include=["Dimension", "Fact", "Integration"],
                schemas_to_exclude=["sys", "INFORMATION_SCHEMA"]
            )
        }
    
    def add_database_config(self, config: DatabaseConfig):
        """Add or update database configuration"""
        self.database_configs[config.name] = config
        logger.info(f"Added database configuration: {config.name}")
    
    def remove_database_config(self, database_name: str):
        """Remove database configuration"""
        if database_name in self.database_configs:
            del self.database_configs[database_name]
            logger.info(f"Removed database configuration: {database_name}")
    
    async def initialize_pipeline(self) -> bool:
        """Initialize all pipeline components"""
        try:
            logger.info("Initializing multi-database RAG pipeline...")
            
            # Initialize embedding client
            embedding_init = await self.embedding_client.initialize()
            if not embedding_init:
                logger.error("Failed to initialize embedding client")
                return False
            
            # Initialize vector store
            vector_init = await self.vector_store.initialize()
            if not vector_init:
                logger.error("Failed to initialize vector store")
                return False
            
            # Setup RAG collections
            collections_init = await self.vector_manager.setup_rag_collections()
            if not collections_init:
                logger.error("Failed to setup RAG collections")
                return False
            
            # Initialize graph store
            graph_init = await self.graph_store.initialize()
            if not graph_init:
                logger.error("Failed to initialize graph store")
                return False
            
            # Initialize dbt integration
            dbt_init = await self.dbt_integration.initialize()
            if not dbt_init:
                logger.error("Failed to initialize dbt integration")
                return False
            
            logger.info("Pipeline initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            return False
    
    async def discover_all_databases(self) -> Dict[str, Dict]:
        """Discover schemas for all configured databases"""
        all_discoveries = {}
        
        for db_name, config in self.database_configs.items():
            if not config.enabled:
                continue
                
            logger.info(f"Discovering schema for database: {db_name}")
            try:
                schema = await self.discovery.discover_database_schema(db_name)
                all_discoveries[db_name] = schema
                logger.info(f"Discovered {len(schema)} tables in {db_name}")
                
            except Exception as e:
                logger.error(f"Failed to discover schema for {db_name}: {e}")
                all_discoveries[db_name] = {}
        
        return all_discoveries
    
    async def process_database(self, database_name: str, discovery_data: Dict = None) -> ProcessingStats:
        """Process a single database through the RAG pipeline"""
        config = self.database_configs.get(database_name)
        if not config or not config.enabled:
            logger.warning(f"Database {database_name} is not configured or disabled")
            return ProcessingStats(database_name, datetime.now())
        
        stats = ProcessingStats(database_name, datetime.now())
        
        try:
            logger.info(f"Processing database: {database_name}")
            
            # Get database schema if not provided
            if discovery_data is None:
                discovery_data = await self.discovery.discover_database_schema(database_name)
            
            # Filter tables based on configuration
            filtered_tables = self._filter_tables(discovery_data, config)
            stats.total_tables = len(filtered_tables)
            
            logger.info(f"Processing {len(filtered_tables)} tables from {database_name}")
            
            # Process each table
            for table_name, table_info in filtered_tables.items():
                try:
                    await self._process_table(table_info, config, stats)
                    stats.processed_tables += 1
                    
                except Exception as e:
                    error_msg = f"Failed to process table {table_name}: {e}"
                    logger.error(error_msg)
                    stats.errors.append(error_msg)
            
            stats.end_time = datetime.now()
            logger.info(f"Completed processing {database_name}: {stats.processed_tables}/{stats.total_tables} tables, "
                       f"{stats.successful_embeddings} embeddings, {stats.success_rate:.1f}% success rate")
            
            return stats
            
        except Exception as e:
            stats.end_time = datetime.now()
            error_msg = f"Database processing failed for {database_name}: {e}"
            logger.error(error_msg)
            stats.errors.append(error_msg)
            return stats
    
    def _filter_tables(self, discovery_data: Dict, config: DatabaseConfig) -> Dict:
        """Filter tables based on configuration"""
        filtered = {}
        
        for table_name, table_info in discovery_data.items():
            schema_name = table_info.schema_name
            
            # Check schema inclusion/exclusion
            if config.schemas_to_include and schema_name not in config.schemas_to_include:
                continue
            if config.schemas_to_exclude and schema_name in config.schemas_to_exclude:
                continue
            
            # Check table inclusion/exclusion
            if config.tables_to_include and table_name not in config.tables_to_include:
                continue
            if config.tables_to_exclude and table_name in config.tables_to_exclude:
                continue
            
            filtered[table_name] = table_info
        
        return filtered
    
    async def _process_table(self, table_info, config: DatabaseConfig, stats: ProcessingStats):
        """Process a single table through the RAG pipeline"""
        try:
            # Extract table data (simplified for demo)
            table_data = await self._extract_sample_table_data(
                table_info,
                config.sample_limit
            )
            
            if not table_data:
                logger.warning(f"No data extracted from {table_info.schema_name}.{table_info.table_name}")
                return
            
            stats.total_records += len(table_data)
            
            # Get dbt semantic information
            dbt_info = await self.dbt_integration.get_table_semantic_info(
                schema=table_info.schema_name,
                table=table_info.table_name
            )
            
            # Create enhanced text representations
            enhanced_texts = []
            for record in table_data:
                enhanced_text = self._create_enhanced_text(record, table_info, dbt_info)
                enhanced_texts.append(enhanced_text)
            
            # Generate embeddings
            embeddings = await self.embedding_client.embed_text(
                enhanced_texts,
                task="retrieval.passage"
            )
            
            if not embeddings:
                logger.error(f"Failed to generate embeddings for {table_info.schema_name}.{table_info.table_name}")
                return
            
            stats.successful_embeddings += len(embeddings)
            
            # Store in vector database
            vector_success = await self.vector_manager.ingest_sql_data(
                table_data=table_data,
                embeddings=embeddings,
                database_name=table_info.database_name,
                schema_name=table_info.schema_name,
                table_name=table_info.table_name,
                collection_name="documents"
            )
            
            if vector_success:
                stats.vector_insertions += len(embeddings)
            
            # Store relationships in graph database
            graph_success = await self._store_graph_relationships(table_info, table_data, dbt_info)
            if graph_success:
                stats.graph_insertions += len(table_data)
            
            logger.info(f"Processed {table_info.schema_name}.{table_info.table_name}: "
                       f"{len(table_data)} records, {len(embeddings)} embeddings")
            
        except Exception as e:
            logger.error(f"Failed to process table {table_info.schema_name}.{table_info.table_name}: {e}")
            raise
    
    def _create_enhanced_text(self, record: Dict, table_info, dbt_info: Dict) -> str:
        """Create enhanced text representation with semantic context"""
        try:
            # Base record information
            text_parts = []
            
            # Add database context
            text_parts.append(f"Database: {table_info.database_name}")
            text_parts.append(f"Schema: {table_info.schema_name}")
            text_parts.append(f"Table: {table_info.table_name}")
            
            # Add dbt semantic information
            if dbt_info:
                if dbt_info.description:
                    text_parts.append(f"Description: {dbt_info.description}")
                
                if dbt_info.business_rules:
                    text_parts.append(f"Business Rules: {'; '.join(dbt_info.business_rules)}")
            
            # Add record data
            record_parts = []
            for key, value in record.items():
                if value is not None:
                    # Add semantic column context if available
                    column_context = ""
                    if dbt_info and dbt_info.columns:
                        for col_info in dbt_info.columns:
                            if col_info.name == key and col_info.description:
                                column_context = f" ({col_info.description})"
                                break
                    
                    record_parts.append(f"{key}{column_context}: {value}")
            
            text_parts.append("Data: " + " | ".join(record_parts))
            
            return " • ".join(text_parts)
            
        except Exception as e:
            logger.error(f"Failed to create enhanced text: {e}")
            return str(record)
    
    async def _extract_sample_table_data(self, table_info, limit: int) -> List[Dict]:
        """Extract sample data from a table"""
        try:
            import pyodbc
            
            # Build connection string
            settings = get_settings()
            connection_string = (
                f"DRIVER={{{settings.db_driver}}};"
                f"SERVER={settings.db_server},{settings.db_port};"
                f"DATABASE={table_info.database_name};"
                f"UID={settings.db_username};"
                f"PWD={settings.db_password};"
                f"TrustServerCertificate=yes;"
            )
            
            # Connect and query
            conn = pyodbc.connect(connection_string)
            cursor = conn.cursor()
            
            # Query with limit
            query = f"SELECT TOP {limit} * FROM [{table_info.schema_name}].[{table_info.table_name}]"
            cursor.execute(query)
            
            # Get column names
            columns = [desc[0] for desc in cursor.description]
            
            # Fetch data
            rows = cursor.fetchall()
            
            # Convert to dict list
            table_data = []
            for row in rows:
                record = {}
                for i, value in enumerate(row):
                    # Handle unsupported SQL types
                    if value is None:
                        record[columns[i]] = None
                    else:
                        try:
                            record[columns[i]] = str(value)
                        except:
                            record[columns[i]] = f"<unsupported_type_{type(value).__name__}>"
                table_data.append(record)
            
            conn.close()
            return table_data
            
        except Exception as e:
            logger.error(f"Failed to extract table data: {e}")
            return []
    
    async def _store_graph_relationships(self, table_info, table_data: List[Dict], dbt_info: Dict) -> bool:
        """Store table relationships in graph database"""
        try:
            # Create table node
            table_node = {
                "id": f"{table_info.database_name}.{table_info.schema_name}.{table_info.table_name}",
                "database": table_info.database_name,
                "schema": table_info.schema_name,
                "table_name": table_info.table_name,
                "type": "table",
                "row_count": len(table_data),
                "description": dbt_info.description if dbt_info else None
            }
            
            await self.graph_store.create_node("Table", table_node)
            
            # Create schema node and relationship
            schema_node = {
                "id": f"{table_info.database_name}.{table_info.schema_name}",
                "database": table_info.database_name,
                "schema": table_info.schema_name,
                "type": "schema"
            }
            
            await self.graph_store.create_node("Schema", schema_node)
            await self.graph_store.create_relationship(
                "Schema", schema_node["id"],
                "Table", table_node["id"],
                "CONTAINS"
            )
            
            # Create database node and relationship
            db_node = {
                "id": table_info.database_name,
                "database": table_info.database_name,
                "type": "database"
            }
            
            await self.graph_store.create_node("Database", db_node)
            await self.graph_store.create_relationship(
                "Database", db_node["id"],
                "Schema", schema_node["id"],
                "CONTAINS"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store graph relationships: {e}")
            return False
    
    async def process_all_databases(self, discovery_data: Dict = None) -> Dict[str, ProcessingStats]:
        """Process all configured databases"""
        all_stats = {}
        
        # Discover all databases if not provided
        if discovery_data is None:
            discovery_data = await self.discover_all_databases()
        
        # Sort databases by priority
        sorted_databases = sorted(
            [(name, config) for name, config in self.database_configs.items() if config.enabled],
            key=lambda x: x[1].priority
        )
        
        for database_name, config in sorted_databases:
            db_discovery = discovery_data.get(database_name, {})
            stats = await self.process_database(database_name, db_discovery)
            all_stats[database_name] = stats
        
        return all_stats
    
    async def get_processing_summary(self, stats: Dict[str, ProcessingStats]) -> Dict:
        """Generate processing summary"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_databases": len(stats),
            "total_tables": sum(s.total_tables for s in stats.values()),
            "processed_tables": sum(s.processed_tables for s in stats.values()),
            "total_records": sum(s.total_records for s in stats.values()),
            "successful_embeddings": sum(s.successful_embeddings for s in stats.values()),
            "vector_insertions": sum(s.vector_insertions for s in stats.values()),
            "graph_insertions": sum(s.graph_insertions for s in stats.values()),
            "total_errors": sum(len(s.errors) for s in stats.values()),
            "processing_time": sum(s.duration for s in stats.values()),
            "databases": {}
        }
        
        for db_name, db_stats in stats.items():
            summary["databases"][db_name] = {
                "tables": f"{db_stats.processed_tables}/{db_stats.total_tables}",
                "records": db_stats.total_records,
                "embeddings": db_stats.successful_embeddings,
                "success_rate": f"{db_stats.success_rate:.1f}%",
                "duration": f"{db_stats.duration:.1f}s",
                "errors": len(db_stats.errors)
            }
        
        return summary
    
    async def export_processing_results(self, stats: Dict[str, ProcessingStats], 
                                      output_path: str = None) -> str:
        """Export processing results to JSON"""
        try:
            if output_path is None:
                output_path = f"multi_database_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Create summary
            summary = await self.get_processing_summary(stats)
            
            # Add detailed stats
            detailed_stats = {}
            for db_name, db_stats in stats.items():
                detailed_stats[db_name] = {
                    "start_time": db_stats.start_time.isoformat(),
                    "end_time": db_stats.end_time.isoformat() if db_stats.end_time else None,
                    "duration": db_stats.duration,
                    "total_tables": db_stats.total_tables,
                    "processed_tables": db_stats.processed_tables,
                    "total_records": db_stats.total_records,
                    "successful_embeddings": db_stats.successful_embeddings,
                    "failed_embeddings": db_stats.failed_embeddings,
                    "vector_insertions": db_stats.vector_insertions,
                    "graph_insertions": db_stats.graph_insertions,
                    "success_rate": db_stats.success_rate,
                    "errors": db_stats.errors
                }
            
            results = {
                "summary": summary,
                "detailed_stats": detailed_stats,
                "database_configs": {
                    name: {
                        "enabled": config.enabled,
                        "priority": config.priority,
                        "sample_limit": config.sample_limit,
                        "schemas_to_include": config.schemas_to_include,
                        "schemas_to_exclude": config.schemas_to_exclude
                    }
                    for name, config in self.database_configs.items()
                }
            }
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Processing results exported to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export processing results: {e}")
            return None
    
    async def shutdown(self):
        """Shutdown all pipeline components"""
        try:
            logger.info("Shutting down multi-database pipeline...")
            
            # Shutdown components
            if hasattr(self.vector_store, 'shutdown'):
                await self.vector_store.shutdown()
            
            if hasattr(self.graph_store, 'shutdown'):
                await self.graph_store.shutdown()
            
            if hasattr(self.embedding_client, 'shutdown'):
                await self.embedding_client.shutdown()
            
            logger.info("Pipeline shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during pipeline shutdown: {e}")


async def main():
    """Example usage of multi-database pipeline"""
    pipeline = MultiDatabasePipeline()
    
    try:
        # Initialize pipeline
        if not await pipeline.initialize_pipeline():
            logger.error("Failed to initialize pipeline")
            return
        
        # Process all databases
        stats = await pipeline.process_all_databases()
        
        # Export results
        results_file = await pipeline.export_processing_results(stats)
        
        # Print summary
        summary = await pipeline.get_processing_summary(stats)
        print(f"\\nMulti-Database RAG Pipeline Summary:")
        print(f"Processed {summary['total_databases']} databases")
        print(f"Tables: {summary['processed_tables']}/{summary['total_tables']}")
        print(f"Records: {summary['total_records']:,}")
        print(f"Embeddings: {summary['successful_embeddings']:,}")
        print(f"Vector insertions: {summary['vector_insertions']:,}")
        print(f"Graph insertions: {summary['graph_insertions']:,}")
        print(f"Total errors: {summary['total_errors']}")
        print(f"Processing time: {summary['processing_time']:.1f}s")
        
        if results_file:
            print(f"\\nDetailed results exported to: {results_file}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        
    finally:
        await pipeline.shutdown()


if __name__ == "__main__":
    asyncio.run(main())