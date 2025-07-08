"""
Database Discovery Tool
======================

Automated database schema discovery for multi-database RAG systems.
Supports both WideWorldImporters and WideWorldImportersDW databases.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pyodbc
from datetime import datetime

from ..config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class TableInfo:
    """Information about a database table"""
    database_name: str
    schema_name: str
    table_name: str
    table_type: str
    row_count: Optional[int] = None
    columns: List[Dict[str, Any]] = None
    primary_keys: List[str] = None
    foreign_keys: List[Dict[str, Any]] = None
    indexes: List[Dict[str, Any]] = None
    constraints: List[Dict[str, Any]] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        if self.columns is None:
            self.columns = []
        if self.primary_keys is None:
            self.primary_keys = []
        if self.foreign_keys is None:
            self.foreign_keys = []
        if self.indexes is None:
            self.indexes = []
        if self.constraints is None:
            self.constraints = []


@dataclass
class ColumnInfo:
    """Information about a database column"""
    column_name: str
    data_type: str
    is_nullable: bool
    default_value: Optional[str] = None
    max_length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None
    is_identity: bool = False
    is_computed: bool = False
    description: Optional[str] = None


class DatabaseDiscovery:
    """Database schema discovery and metadata extraction"""
    
    def __init__(self, 
                 server: str = None,
                 port: int = None,
                 username: str = None,
                 password: str = None,
                 driver: str = None):
        settings = get_settings()
        
        self.server = server or settings.db_server
        self.port = port or settings.db_port
        self.username = username or settings.db_username
        self.password = password or settings.db_password
        self.driver = driver or settings.db_driver
        
        self.connection = None
        self.discovered_databases = {}
        
    async def connect(self, database_name: str) -> bool:
        """Connect to a specific database"""
        try:
            connection_string = (
                f"DRIVER={{{self.driver}}};"
                f"SERVER={self.server},{self.port};"
                f"DATABASE={database_name};"
                f"UID={self.username};"
                f"PWD={self.password};"
                f"TrustServerCertificate=yes;"
            )
            
            self.connection = pyodbc.connect(connection_string)
            logger.info(f"Connected to database: {database_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to {database_name}: {e}")
            return False
    
    async def discover_database_schema(self, database_name: str) -> Dict[str, TableInfo]:
        """Discover complete schema for a database"""
        try:
            if not await self.connect(database_name):
                return {}
            
            cursor = self.connection.cursor()
            tables = {}
            
            # Get all tables and views
            table_query = """
            SELECT 
                TABLE_CATALOG,
                TABLE_SCHEMA,
                TABLE_NAME,
                TABLE_TYPE
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE IN ('BASE TABLE', 'VIEW')
            ORDER BY TABLE_SCHEMA, TABLE_NAME
            """
            
            cursor.execute(table_query)
            table_rows = cursor.fetchall()
            
            for row in table_rows:
                table_info = TableInfo(
                    database_name=row.TABLE_CATALOG,
                    schema_name=row.TABLE_SCHEMA,
                    table_name=row.TABLE_NAME,
                    table_type=row.TABLE_TYPE
                )
                
                # Get detailed information for each table
                await self._get_table_details(cursor, table_info)
                
                full_table_name = f"{table_info.schema_name}.{table_info.table_name}"
                tables[full_table_name] = table_info
                
            logger.info(f"Discovered {len(tables)} tables in {database_name}")
            return tables
            
        except Exception as e:
            logger.error(f"Failed to discover schema for {database_name}: {e}")
            return {}
        finally:
            if self.connection:
                self.connection.close()
    
    async def _get_table_details(self, cursor, table_info: TableInfo):
        """Get detailed information about a table"""
        try:
            # Get columns
            await self._get_column_info(cursor, table_info)
            
            # Get primary keys
            await self._get_primary_keys(cursor, table_info)
            
            # Get foreign keys
            await self._get_foreign_keys(cursor, table_info)
            
            # Get indexes
            await self._get_indexes(cursor, table_info)
            
            # Get constraints
            await self._get_constraints(cursor, table_info)
            
            # Get row count (for base tables only)
            if table_info.table_type == 'BASE TABLE':
                await self._get_row_count(cursor, table_info)
                
        except Exception as e:
            logger.error(f"Failed to get details for {table_info.table_name}: {e}")
    
    async def _get_column_info(self, cursor, table_info: TableInfo):
        """Get column information"""
        try:
            column_query = """
            SELECT 
                COLUMN_NAME,
                DATA_TYPE,
                IS_NULLABLE,
                COLUMN_DEFAULT,
                CHARACTER_MAXIMUM_LENGTH,
                NUMERIC_PRECISION,
                NUMERIC_SCALE,
                COLUMNPROPERTY(OBJECT_ID(TABLE_SCHEMA + '.' + TABLE_NAME), COLUMN_NAME, 'IsIdentity') as IS_IDENTITY,
                COLUMNPROPERTY(OBJECT_ID(TABLE_SCHEMA + '.' + TABLE_NAME), COLUMN_NAME, 'IsComputed') as IS_COMPUTED
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
            ORDER BY ORDINAL_POSITION
            """
            
            cursor.execute(column_query, table_info.schema_name, table_info.table_name)
            columns = cursor.fetchall()
            
            for col in columns:
                column_info = {
                    'column_name': col.COLUMN_NAME,
                    'data_type': col.DATA_TYPE,
                    'is_nullable': col.IS_NULLABLE == 'YES',
                    'default_value': col.COLUMN_DEFAULT,
                    'max_length': col.CHARACTER_MAXIMUM_LENGTH,
                    'precision': col.NUMERIC_PRECISION,
                    'scale': col.NUMERIC_SCALE,
                    'is_identity': bool(col.IS_IDENTITY),
                    'is_computed': bool(col.IS_COMPUTED)
                }
                table_info.columns.append(column_info)
                
        except Exception as e:
            logger.error(f"Failed to get columns for {table_info.table_name}: {e}")
    
    async def _get_primary_keys(self, cursor, table_info: TableInfo):
        """Get primary key information"""
        try:
            pk_query = """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
            AND CONSTRAINT_NAME IN (
                SELECT CONSTRAINT_NAME
                FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS
                WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? AND CONSTRAINT_TYPE = 'PRIMARY KEY'
            )
            ORDER BY ORDINAL_POSITION
            """
            
            cursor.execute(pk_query, table_info.schema_name, table_info.table_name,
                          table_info.schema_name, table_info.table_name)
            pks = cursor.fetchall()
            
            table_info.primary_keys = [pk.COLUMN_NAME for pk in pks]
            
        except Exception as e:
            logger.error(f"Failed to get primary keys for {table_info.table_name}: {e}")
    
    async def _get_foreign_keys(self, cursor, table_info: TableInfo):
        """Get foreign key information"""
        try:
            fk_query = """
            SELECT 
                kcu.COLUMN_NAME,
                kcu.CONSTRAINT_NAME,
                rcu.TABLE_SCHEMA as REFERENCED_SCHEMA,
                rcu.TABLE_NAME as REFERENCED_TABLE,
                rcu.COLUMN_NAME as REFERENCED_COLUMN
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
            JOIN INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc 
                ON kcu.CONSTRAINT_NAME = rc.CONSTRAINT_NAME
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE rcu 
                ON rc.UNIQUE_CONSTRAINT_NAME = rcu.CONSTRAINT_NAME
            WHERE kcu.TABLE_SCHEMA = ? AND kcu.TABLE_NAME = ?
            ORDER BY kcu.ORDINAL_POSITION
            """
            
            cursor.execute(fk_query, table_info.schema_name, table_info.table_name)
            fks = cursor.fetchall()
            
            for fk in fks:
                fk_info = {
                    'column_name': fk.COLUMN_NAME,
                    'constraint_name': fk.CONSTRAINT_NAME,
                    'referenced_schema': fk.REFERENCED_SCHEMA,
                    'referenced_table': fk.REFERENCED_TABLE,
                    'referenced_column': fk.REFERENCED_COLUMN
                }
                table_info.foreign_keys.append(fk_info)
                
        except Exception as e:
            logger.error(f"Failed to get foreign keys for {table_info.table_name}: {e}")
    
    async def _get_indexes(self, cursor, table_info: TableInfo):
        """Get index information"""
        try:
            index_query = """
            SELECT 
                i.name as INDEX_NAME,
                i.type_desc as INDEX_TYPE,
                i.is_unique,
                i.is_primary_key,
                STRING_AGG(c.name, ', ') as COLUMNS
            FROM sys.indexes i
            JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
            JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
            WHERE i.object_id = OBJECT_ID(? + '.' + ?)
            GROUP BY i.name, i.type_desc, i.is_unique, i.is_primary_key
            """
            
            full_table_name = f"{table_info.schema_name}.{table_info.table_name}"
            cursor.execute(index_query, table_info.schema_name, table_info.table_name)
            indexes = cursor.fetchall()
            
            for idx in indexes:
                index_info = {
                    'index_name': idx.INDEX_NAME,
                    'index_type': idx.INDEX_TYPE,
                    'is_unique': bool(idx.is_unique),
                    'is_primary_key': bool(idx.is_primary_key),
                    'columns': idx.COLUMNS
                }
                table_info.indexes.append(index_info)
                
        except Exception as e:
            logger.error(f"Failed to get indexes for {table_info.table_name}: {e}")
    
    async def _get_constraints(self, cursor, table_info: TableInfo):
        """Get constraint information"""
        try:
            constraint_query = """
            SELECT 
                tc.CONSTRAINT_NAME,
                tc.CONSTRAINT_TYPE,
                kcu.COLUMN_NAME,
                cc.CHECK_CLAUSE
            FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
            LEFT JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu 
                ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
            LEFT JOIN INFORMATION_SCHEMA.CHECK_CONSTRAINTS cc 
                ON tc.CONSTRAINT_NAME = cc.CONSTRAINT_NAME
            WHERE tc.TABLE_SCHEMA = ? AND tc.TABLE_NAME = ?
            """
            
            cursor.execute(constraint_query, table_info.schema_name, table_info.table_name)
            constraints = cursor.fetchall()
            
            for constraint in constraints:
                constraint_info = {
                    'constraint_name': constraint.CONSTRAINT_NAME,
                    'constraint_type': constraint.CONSTRAINT_TYPE,
                    'column_name': constraint.COLUMN_NAME,
                    'check_clause': constraint.CHECK_CLAUSE
                }
                table_info.constraints.append(constraint_info)
                
        except Exception as e:
            logger.error(f"Failed to get constraints for {table_info.table_name}: {e}")
    
    async def _get_row_count(self, cursor, table_info: TableInfo):
        """Get approximate row count"""
        try:
            count_query = f"""
            SELECT COUNT(*)
            FROM [{table_info.schema_name}].[{table_info.table_name}]
            """
            
            cursor.execute(count_query)
            result = cursor.fetchone()
            table_info.row_count = result[0] if result else 0
            
        except Exception as e:
            logger.error(f"Failed to get row count for {table_info.table_name}: {e}")
            table_info.row_count = None
    
    async def discover_multiple_databases(self, databases: List[str]) -> Dict[str, Dict[str, TableInfo]]:
        """Discover schemas for multiple databases"""
        all_discoveries = {}
        
        for database in databases:
            logger.info(f"Discovering schema for database: {database}")
            schema = await self.discover_database_schema(database)
            all_discoveries[database] = schema
            
        return all_discoveries
    
    async def export_discovery_to_json(self, discovery_data: Dict, output_path: str = None):
        """Export discovery results to JSON file"""
        try:
            if output_path is None:
                output_path = f"database_discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert TableInfo objects to dictionaries
            serializable_data = {}
            for db_name, tables in discovery_data.items():
                serializable_data[db_name] = {}
                for table_name, table_info in tables.items():
                    serializable_data[db_name][table_name] = asdict(table_info)
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, default=str)
            
            logger.info(f"Discovery results exported to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export discovery results: {e}")
            return None
    
    async def get_database_relationships(self, discovery_data: Dict) -> Dict[str, List[Dict]]:
        """Analyze relationships between databases"""
        relationships = {}
        
        try:
            for db_name, tables in discovery_data.items():
                db_relationships = []
                
                for table_name, table_info in tables.items():
                    # Look for foreign key relationships
                    for fk in table_info.foreign_keys:
                        relationship = {
                            'source_database': db_name,
                            'source_table': table_name,
                            'source_column': fk['column_name'],
                            'target_schema': fk['referenced_schema'],
                            'target_table': fk['referenced_table'],
                            'target_column': fk['referenced_column'],
                            'relationship_type': 'foreign_key'
                        }
                        db_relationships.append(relationship)
                
                relationships[db_name] = db_relationships
            
            return relationships
            
        except Exception as e:
            logger.error(f"Failed to analyze relationships: {e}")
            return {}
    
    async def generate_dbt_sources_yaml(self, discovery_data: Dict, output_path: str = None) -> str:
        """Generate dbt sources.yml from discovery data"""
        try:
            if output_path is None:
                output_path = "generated_sources.yml"
            
            yaml_content = ["version: 2", "", "sources:"]
            
            for db_name, tables in discovery_data.items():
                # Group tables by schema
                schemas = {}
                for table_name, table_info in tables.items():
                    schema = table_info.schema_name
                    if schema not in schemas:
                        schemas[schema] = []
                    schemas[schema].append(table_info)
                
                # Generate YAML for each schema
                for schema_name, schema_tables in schemas.items():
                    yaml_content.extend([
                        f"  # {db_name}",
                        f"  - name: {schema_name.lower()}",
                        f"    database: {db_name}",
                        f"    schema: {schema_name}",
                        "    tables:"
                    ])
                    
                    for table_info in schema_tables:
                        yaml_content.append(f"      - name: {table_info.table_name}")
                        if table_info.description:
                            yaml_content.append(f"        description: {table_info.description}")
                    
                    yaml_content.append("")
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yaml_content))
            
            logger.info(f"Generated dbt sources.yml at: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to generate dbt sources.yml: {e}")
            return None


async def main():
    """Example usage of database discovery"""
    discovery = DatabaseDiscovery()
    
    # Discover both databases
    databases = ['WideWorldImporters', 'WideWorldImportersDW']
    all_discoveries = await discovery.discover_multiple_databases(databases)
    
    # Export results
    await discovery.export_discovery_to_json(all_discoveries)
    
    # Generate dbt sources
    await discovery.generate_dbt_sources_yaml(all_discoveries)
    
    # Analyze relationships
    relationships = await discovery.get_database_relationships(all_discoveries)
    
    print("Database Discovery Complete!")
    for db_name, tables in all_discoveries.items():
        print(f"{db_name}: {len(tables)} tables discovered")


if __name__ == "__main__":
    asyncio.run(main())