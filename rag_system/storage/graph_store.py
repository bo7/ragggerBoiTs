"""
Neo4j Graph Store Integration
=============================

Neo4j graph database integration for storing and querying relationships
in the multi-database RAG system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import json

from ..config import get_settings

logger = logging.getLogger(__name__)


class Neo4jGraphStore:
    """Neo4j graph database client for RAG system"""
    
    def __init__(self, 
                 uri: str = None,
                 username: str = None,
                 password: str = None,
                 database: str = None):
        self.uri = uri or "bolt://localhost:7687"
        self.username = username or "neo4j"
        self.password = password or "password123"
        self.database = database or "neo4j"
        
        # Driver will be initialized lazily
        self.driver = None
        self.is_connected = False
        
    async def initialize(self) -> bool:
        """Initialize connection to Neo4j"""
        try:
            # Import Neo4j driver
            from neo4j import GraphDatabase
            
            # Create driver
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            
            # Test connection
            await self.verify_connectivity()
            
            self.is_connected = True
            logger.info(f"Connected to Neo4j at {self.uri}")
            return True
            
        except ImportError:
            logger.error("neo4j not installed. Install with: pip install neo4j")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    async def verify_connectivity(self):
        """Verify connection to Neo4j"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record["test"] != 1:
                    raise Exception("Connection test failed")
        except Exception as e:
            raise Exception(f"Neo4j connectivity test failed: {e}")
    
    async def create_node(self, label: str, properties: Dict[str, Any]) -> bool:
        """Create a node in the graph"""
        try:
            with self.driver.session(database=self.database) as session:
                # Build property string
                props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
                
                query = f"""
                MERGE (n:{label} {{id: $id}})
                SET n += {{{props_str}}}
                RETURN n
                """
                
                result = session.run(query, **properties)
                record = result.single()
                
                if record:
                    logger.debug(f"Created/updated {label} node: {properties.get('id', 'unknown')}")
                    return True
                else:
                    logger.error(f"Failed to create {label} node")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to create node {label}: {e}")
            return False
    
    async def create_relationship(self, 
                                from_label: str, 
                                from_id: str,
                                to_label: str,
                                to_id: str,
                                relationship_type: str,
                                properties: Dict[str, Any] = None) -> bool:
        """Create a relationship between two nodes"""
        try:
            with self.driver.session(database=self.database) as session:
                # Build relationship properties
                rel_props = ""
                if properties:
                    props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
                    rel_props = f" {{{props_str}}}"
                
                query = f"""
                MATCH (from:{from_label} {{id: $from_id}})
                MATCH (to:{to_label} {{id: $to_id}})
                MERGE (from)-[r:{relationship_type}]->(to)
                SET r += {{{", ".join([f"{k}: ${k}" for k in (properties or {}).keys()])}}}
                RETURN r
                """
                
                params = {
                    "from_id": from_id,
                    "to_id": to_id,
                    **(properties or {})
                }
                
                result = session.run(query, **params)
                record = result.single()
                
                if record:
                    logger.debug(f"Created {relationship_type} relationship: {from_id} -> {to_id}")
                    return True
                else:
                    logger.error(f"Failed to create {relationship_type} relationship")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to create relationship {relationship_type}: {e}")
            return False
    
    async def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict]:
        """Execute a custom Cypher query"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters or {})
                records = []
                
                for record in result:
                    # Convert neo4j record to dictionary
                    record_dict = {}
                    for key in record.keys():
                        value = record[key]
                        # Handle Neo4j types
                        if hasattr(value, '__dict__'):
                            record_dict[key] = dict(value)
                        else:
                            record_dict[key] = value
                    records.append(record_dict)
                
                return records
                
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            return []
    
    async def find_nodes(self, 
                        label: str, 
                        properties: Dict[str, Any] = None,
                        limit: int = 100) -> List[Dict]:
        """Find nodes matching criteria"""
        try:
            where_clause = ""
            params = {}
            
            if properties:
                conditions = []
                for key, value in properties.items():
                    conditions.append(f"n.{key} = ${key}")
                    params[key] = value
                where_clause = f"WHERE {' AND '.join(conditions)}"
            
            query = f"""
            MATCH (n:{label})
            {where_clause}
            RETURN n
            LIMIT {limit}
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, params)
                nodes = []
                
                for record in result:
                    node = dict(record["n"])
                    nodes.append(node)
                
                return nodes
                
        except Exception as e:
            logger.error(f"Failed to find nodes: {e}")
            return []
    
    async def find_relationships(self, 
                               from_label: str = None,
                               to_label: str = None,
                               relationship_type: str = None,
                               properties: Dict[str, Any] = None,
                               limit: int = 100) -> List[Dict]:
        """Find relationships matching criteria"""
        try:
            # Build match pattern
            from_part = f"({from_label})" if from_label else "()"
            to_part = f"({to_label})" if to_label else "()"
            rel_part = f"[r:{relationship_type}]" if relationship_type else "[r]"
            
            where_conditions = []
            params = {}
            
            if properties:
                for key, value in properties.items():
                    where_conditions.append(f"r.{key} = ${key}")
                    params[key] = value
            
            where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
            
            query = f"""
            MATCH {from_part}-{rel_part}->{to_part}
            {where_clause}
            RETURN startNode(r) as from_node, r as relationship, endNode(r) as to_node
            LIMIT {limit}
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, params)
                relationships = []
                
                for record in result:
                    rel_data = {
                        "from_node": dict(record["from_node"]),
                        "relationship": dict(record["relationship"]),
                        "to_node": dict(record["to_node"])
                    }
                    relationships.append(rel_data)
                
                return relationships
                
        except Exception as e:
            logger.error(f"Failed to find relationships: {e}")
            return []
    
    async def get_node_neighbors(self, 
                               label: str, 
                               node_id: str,
                               relationship_type: str = None,
                               direction: str = "both",
                               limit: int = 50) -> List[Dict]:
        """Get neighboring nodes of a specific node"""
        try:
            # Build relationship pattern based on direction
            if direction == "outgoing":
                rel_pattern = f"-[r{':' + relationship_type if relationship_type else ''}]->"
            elif direction == "incoming":
                rel_pattern = f"<-[r{':' + relationship_type if relationship_type else ''}]-"
            else:  # both
                rel_pattern = f"-[r{':' + relationship_type if relationship_type else ''}]-"
            
            query = f"""
            MATCH (n:{label} {{id: $node_id}}){rel_pattern}(neighbor)
            RETURN neighbor, r, labels(neighbor) as neighbor_labels
            LIMIT {limit}
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, {"node_id": node_id})
                neighbors = []
                
                for record in result:
                    neighbor_data = {
                        "node": dict(record["neighbor"]),
                        "relationship": dict(record["r"]),
                        "labels": record["neighbor_labels"]
                    }
                    neighbors.append(neighbor_data)
                
                return neighbors
                
        except Exception as e:
            logger.error(f"Failed to get neighbors: {e}")
            return []
    
    async def get_shortest_path(self, 
                              from_label: str, 
                              from_id: str,
                              to_label: str, 
                              to_id: str,
                              relationship_type: str = None,
                              max_length: int = 6) -> List[Dict]:
        """Find shortest path between two nodes"""
        try:
            rel_filter = f":{relationship_type}" if relationship_type else ""
            
            query = f"""
            MATCH path = shortestPath((from:{from_label} {{id: $from_id}})-[{rel_filter}*1..{max_length}]-(to:{to_label} {{id: $to_id}}))
            RETURN path, length(path) as path_length
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, {"from_id": from_id, "to_id": to_id})
                paths = []
                
                for record in result:
                    path = record["path"]
                    path_data = {
                        "length": record["path_length"],
                        "nodes": [dict(node) for node in path.nodes],
                        "relationships": [dict(rel) for rel in path.relationships]
                    }
                    paths.append(path_data)
                
                return paths
                
        except Exception as e:
            logger.error(f"Failed to find shortest path: {e}")
            return []
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            queries = {
                "node_count": "MATCH (n) RETURN count(n) as count",
                "relationship_count": "MATCH ()-[r]->() RETURN count(r) as count",
                "node_labels": "CALL db.labels() YIELD label RETURN collect(label) as labels",
                "relationship_types": "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types"
            }
            
            stats = {}
            
            with self.driver.session(database=self.database) as session:
                for stat_name, query in queries.items():
                    try:
                        result = session.run(query)
                        record = result.single()
                        if stat_name in ["node_labels", "relationship_types"]:
                            stats[stat_name] = record[list(record.keys())[0]]
                        else:
                            stats[stat_name] = record["count"]
                    except Exception as e:
                        logger.warning(f"Failed to get {stat_name}: {e}")
                        stats[stat_name] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    async def clear_database(self) -> bool:
        """Clear all data from the database"""
        try:
            with self.driver.session(database=self.database) as session:
                # Delete all relationships first
                session.run("MATCH ()-[r]->() DELETE r")
                
                # Then delete all nodes
                session.run("MATCH (n) DELETE n")
                
                logger.info("Neo4j database cleared")
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            return False
    
    async def create_indexes(self) -> bool:
        """Create useful indexes for the RAG system"""
        try:
            indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (n:Database) ON (n.id)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Schema) ON (n.id)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Table) ON (n.id)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Column) ON (n.id)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Record) ON (n.id)"
            ]
            
            with self.driver.session(database=self.database) as session:
                for index_query in indexes:
                    try:
                        session.run(index_query)
                        logger.debug(f"Created index: {index_query}")
                    except Exception as e:
                        logger.warning(f"Failed to create index: {e}")
            
            logger.info("Neo4j indexes created")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Neo4j connection health"""
        try:
            if not self.is_connected:
                return {"healthy": False, "error": "Not connected"}
            
            # Test basic connectivity
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                
                if record["test"] != 1:
                    return {"healthy": False, "error": "Test query failed"}
            
            # Get database stats
            stats = await self.get_database_stats()
            
            return {
                "healthy": True,
                "uri": self.uri,
                "database": self.database,
                "stats": stats
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def shutdown(self):
        """Close Neo4j connection"""
        try:
            if self.driver:
                self.driver.close()
                self.is_connected = False
                logger.info("Disconnected from Neo4j")
                
        except Exception as e:
            logger.error(f"Error during Neo4j shutdown: {e}")


class GraphStoreManager:
    """High-level manager for graph store operations"""
    
    def __init__(self, graph_store: Neo4jGraphStore):
        self.graph_store = graph_store
    
    async def setup_rag_schema(self) -> bool:
        """Set up the RAG system schema in Neo4j"""
        try:
            # Create indexes
            await self.graph_store.create_indexes()
            
            # Create constraint to ensure unique IDs
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Database) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Schema) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Table) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Column) REQUIRE n.id IS UNIQUE"
            ]
            
            with self.graph_store.driver.session(database=self.graph_store.database) as session:
                for constraint in constraints:
                    try:
                        session.run(constraint)
                        logger.debug(f"Created constraint: {constraint}")
                    except Exception as e:
                        logger.warning(f"Failed to create constraint: {e}")
            
            logger.info("RAG schema setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup RAG schema: {e}")
            return False
    
    async def store_database_metadata(self, 
                                    database_name: str,
                                    schemas: Dict[str, List[str]],
                                    metadata: Dict[str, Any] = None) -> bool:
        """Store database metadata structure"""
        try:
            # Create database node
            db_props = {
                "id": database_name,
                "name": database_name,
                "type": "database",
                "created_at": datetime.now().isoformat(),
                **(metadata or {})
            }
            
            await self.graph_store.create_node("Database", db_props)
            
            # Create schema nodes and relationships
            for schema_name, tables in schemas.items():
                schema_props = {
                    "id": f"{database_name}.{schema_name}",
                    "name": schema_name,
                    "database": database_name,
                    "type": "schema",
                    "table_count": len(tables)
                }
                
                await self.graph_store.create_node("Schema", schema_props)
                await self.graph_store.create_relationship(
                    "Database", database_name,
                    "Schema", f"{database_name}.{schema_name}",
                    "CONTAINS"
                )
                
                # Create table nodes
                for table_name in tables:
                    table_props = {
                        "id": f"{database_name}.{schema_name}.{table_name}",
                        "name": table_name,
                        "database": database_name,
                        "schema": schema_name,
                        "type": "table"
                    }
                    
                    await self.graph_store.create_node("Table", table_props)
                    await self.graph_store.create_relationship(
                        "Schema", f"{database_name}.{schema_name}",
                        "Table", f"{database_name}.{schema_name}.{table_name}",
                        "CONTAINS"
                    )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store database metadata: {e}")
            return False
    
    async def store_table_relationships(self, 
                                      table_relationships: List[Dict[str, Any]]) -> bool:
        """Store foreign key relationships between tables"""
        try:
            for rel in table_relationships:
                # Create relationship between tables
                await self.graph_store.create_relationship(
                    "Table", rel["source_table_id"],
                    "Table", rel["target_table_id"],
                    "REFERENCES",
                    {
                        "constraint_name": rel.get("constraint_name"),
                        "source_column": rel.get("source_column"),
                        "target_column": rel.get("target_column"),
                        "relationship_type": "foreign_key"
                    }
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store table relationships: {e}")
            return False
    
    async def query_database_structure(self, database_name: str = None) -> Dict[str, Any]:
        """Query the database structure"""
        try:
            if database_name:
                query = """
                MATCH (db:Database {id: $database_name})-[:CONTAINS]->(schema:Schema)-[:CONTAINS]->(table:Table)
                RETURN db, schema, table
                ORDER BY schema.name, table.name
                """
                params = {"database_name": database_name}
            else:
                query = """
                MATCH (db:Database)-[:CONTAINS]->(schema:Schema)-[:CONTAINS]->(table:Table)
                RETURN db, schema, table
                ORDER BY db.name, schema.name, table.name
                """
                params = {}
            
            results = await self.graph_store.execute_query(query, params)
            
            # Organize results
            databases = {}
            for result in results:
                db_name = result["db"]["name"]
                schema_name = result["schema"]["name"]
                table_name = result["table"]["name"]
                
                if db_name not in databases:
                    databases[db_name] = {}
                
                if schema_name not in databases[db_name]:
                    databases[db_name][schema_name] = []
                
                databases[db_name][schema_name].append(table_name)
            
            return databases
            
        except Exception as e:
            logger.error(f"Failed to query database structure: {e}")
            return {}
    
    async def find_related_tables(self, 
                                table_id: str,
                                max_depth: int = 3) -> List[Dict[str, Any]]:
        """Find tables related to a specific table"""
        try:
            query = f"""
            MATCH path = (source:Table {{id: $table_id}})-[:REFERENCES*1..{max_depth}]-(related:Table)
            RETURN path, related
            """
            
            results = await self.graph_store.execute_query(query, {"table_id": table_id})
            
            related_tables = []
            for result in results:
                related_tables.append({
                    "table": result["related"],
                    "path_length": len(result["path"]["relationships"])
                })
            
            return related_tables
            
        except Exception as e:
            logger.error(f"Failed to find related tables: {e}")
            return []


async def main():
    """Example usage of Neo4j graph store"""
    graph_store = Neo4jGraphStore()
    
    try:
        # Initialize
        if not await graph_store.initialize():
            logger.error("Failed to initialize graph store")
            return
        
        # Clear database for testing
        await graph_store.clear_database()
        
        # Create some test data
        await graph_store.create_node("Database", {
            "id": "WideWorldImporters",
            "name": "WideWorldImporters",
            "type": "OLTP"
        })
        
        await graph_store.create_node("Schema", {
            "id": "WideWorldImporters.Sales",
            "name": "Sales",
            "database": "WideWorldImporters"
        })
        
        await graph_store.create_relationship(
            "Database", "WideWorldImporters",
            "Schema", "WideWorldImporters.Sales",
            "CONTAINS"
        )
        
        # Query data
        results = await graph_store.execute_query(
            "MATCH (db:Database)-[:CONTAINS]->(schema:Schema) RETURN db, schema"
        )
        
        print(f"Found {len(results)} database-schema relationships")
        
        # Get stats
        stats = await graph_store.get_database_stats()
        print(f"Database stats: {stats}")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        
    finally:
        await graph_store.shutdown()


if __name__ == "__main__":
    asyncio.run(main())