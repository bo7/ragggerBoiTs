"""
Enhanced SQL Extractor
======================

Builds on existing extract_all.py but adds database constraints,
relationships, and dbt semantic integration.
"""

import json
import logging
import pyodbc
import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dotenv import load_dotenv

from .dbt_integration import DBTSemanticExtractor

logger = logging.getLogger(__name__)


class EnhancedSQLExtractor:
    """Enhanced SQL extractor with constraints and semantic information"""
    
    def __init__(self, 
                 dbt_extractor: Optional[DBTSemanticExtractor] = None,
                 include_constraints: bool = True,
                 include_relationships: bool = True,
                 sample_limit: int = None):
        
        # Load environment
        load_dotenv()
        
        # Database connection
        self.conn_str = (
            f"Driver={{{os.getenv('DB_DRIVER')}}};"
            f"Server={os.getenv('DB_SERVER')},{os.getenv('DB_PORT')};"
            f"Database={os.getenv('DB_DATABASE')};"
            f"UID={os.getenv('DB_USERNAME')};"
            f"PWD={os.getenv('DB_PASSWORD')};"
            "Encrypt=yes;TrustServerCertificate=yes;"
        )
        
        self.sample_limit = sample_limit or int(os.getenv('SAMPLE_LIMIT', 5))
        self.include_constraints = include_constraints
        self.include_relationships = include_relationships
        self.dbt_extractor = dbt_extractor
        
        # Connection and cursor
        self.connection = None
        self.cursor = None
        
    async def initialize(self) -> bool:
        """Initialize SQL connection and dbt extractor"""
        try:
            # Connect to database
            self.connection = pyodbc.connect(self.conn_str, timeout=10)
            self.cursor = self.connection.cursor()
            logger.info("Connected to SQL Server successfully")
            
            # Initialize dbt extractor if provided
            if self.dbt_extractor:
                await self.dbt_extractor.initialize()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SQL extractor: {e}")
            return False
    
    async def extract_enhanced_catalog(self) -> Dict[str, Any]:
        """Extract complete database catalog with constraints and semantics"""
        try:
            catalog = {
                "metadata": {
                    "database": os.getenv('DB_DATABASE'),
                    "server": os.getenv('DB_SERVER'),
                    "extraction_timestamp": "",
                    "includes_constraints": self.include_constraints,
                    "includes_relationships": self.include_relationships,
                    "sample_limit": self.sample_limit
                },
                "tables": [],
                "relationships": {},
                "constraints": {},
                "semantic_info": {}
            }
            
            # Get all tables
            tables = await self._get_tables()
            logger.info(f"Found {len(tables)} tables to extract")
            
            for schema, table in tables:
                # Skip system schemas
                if schema.upper() in ("SYS", "INFORMATION_SCHEMA"):
                    continue
                
                logger.info(f"Extracting {schema}.{table}...")
                
                # Get basic table info
                table_info = await self._extract_table_info(schema, table)
                catalog["tables"].append(table_info)
                
                # Get constraints if enabled
                if self.include_constraints:
                    constraints = await self._get_table_constraints(schema, table)
                    catalog["constraints"][f"{schema}.{table}"] = constraints
                
                # Get semantic info from dbt if available
                if self.dbt_extractor:
                    semantic_info = await self.dbt_extractor.get_table_semantic_info(schema, table)
                    catalog["semantic_info"][f"{schema}.{table}"] = {
                        "description": semantic_info.description,
                        "model_type": semantic_info.model_type,
                        "business_rules": semantic_info.business_rules,
                        "dependencies": semantic_info.dependencies
                    }
            
            # Get relationships if enabled
            if self.include_relationships:
                catalog["relationships"] = await self._get_all_relationships()
            
            logger.info(f"Extraction complete: {len(catalog['tables'])} tables processed")
            return catalog
            
        except Exception as e:
            logger.error(f"Failed to extract enhanced catalog: {e}")
            raise
    
    async def _get_tables(self) -> List[Tuple[str, str]]:
        """Get all tables in the database"""
        try:
            self.cursor.execute("""
                SELECT TABLE_SCHEMA, TABLE_NAME
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_TYPE = 'BASE TABLE'
                ORDER BY TABLE_SCHEMA, TABLE_NAME
            """)
            return [(row.TABLE_SCHEMA, row.TABLE_NAME) for row in self.cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Failed to get tables: {e}")
            return []
    
    async def _extract_table_info(self, schema: str, table: str) -> Dict[str, Any]:
        """Extract comprehensive table information"""
        try:
            # Get columns with enhanced metadata
            columns = await self._get_enhanced_columns(schema, table)
            
            # Get sample rows
            sample_rows = await self._get_sample_rows(schema, table, self.sample_limit)
            
            # Get table statistics
            stats = await self._get_table_statistics(schema, table)
            
            table_info = {
                "schema": schema,
                "name": table,
                "full_name": f"{schema}.{table}",
                "columns": columns,
                "sample_rows": sample_rows,
                "statistics": stats,
                "column_count": len(columns),
                "sample_row_count": len(sample_rows)
            }
            
            # Add dbt semantic description if available
            if self.dbt_extractor:
                semantic_info = await self.dbt_extractor.get_table_semantic_info(schema, table)
                table_info["semantic_description"] = await self.dbt_extractor.generate_semantic_description(semantic_info)
                table_info["dbt_model_type"] = semantic_info.model_type
            
            return table_info
            
        except Exception as e:
            logger.error(f"Failed to extract table info for {schema}.{table}: {e}")
            return {
                "schema": schema,
                "name": table,
                "error": str(e),
                "columns": [],
                "sample_rows": []
            }
    
    async def _get_enhanced_columns(self, schema: str, table: str) -> List[Dict[str, Any]]:
        """Get enhanced column information including constraints"""
        try:
            # Basic column information
            self.cursor.execute("""
                SELECT 
                    c.COLUMN_NAME,
                    c.DATA_TYPE,
                    c.IS_NULLABLE,
                    c.COLUMN_DEFAULT,
                    c.CHARACTER_MAXIMUM_LENGTH,
                    c.NUMERIC_PRECISION,
                    c.NUMERIC_SCALE,
                    c.ORDINAL_POSITION
                FROM INFORMATION_SCHEMA.COLUMNS c
                WHERE c.TABLE_SCHEMA = ? AND c.TABLE_NAME = ?
                ORDER BY c.ORDINAL_POSITION
            """, schema, table)
            
            columns = []
            for row in self.cursor.fetchall():
                col_info = {
                    "name": row.COLUMN_NAME,
                    "data_type": row.DATA_TYPE,
                    "is_nullable": (row.IS_NULLABLE == "YES"),
                    "default_value": row.COLUMN_DEFAULT,
                    "max_length": row.CHARACTER_MAXIMUM_LENGTH,
                    "precision": row.NUMERIC_PRECISION,
                    "scale": row.NUMERIC_SCALE,
                    "ordinal_position": row.ORDINAL_POSITION,
                    "constraints": [],
                    "is_primary_key": False,
                    "is_foreign_key": False,
                    "foreign_key_reference": None
                }
                
                # Add constraint information
                if self.include_constraints:
                    constraints = await self._get_column_constraints(schema, table, row.COLUMN_NAME)
                    col_info.update(constraints)
                
                # Add dbt semantic info if available
                if self.dbt_extractor:
                    semantic_info = await self.dbt_extractor.get_table_semantic_info(schema, table)
                    for col in semantic_info.columns:
                        if col.name.lower() == row.COLUMN_NAME.lower():
                            col_info["description"] = col.description
                            col_info["dbt_tests"] = col.tests
                            col_info["business_constraints"] = col.constraints
                            break
                
                columns.append(col_info)
            
            return columns
            
        except Exception as e:
            logger.error(f"Failed to get enhanced columns for {schema}.{table}: {e}")
            return []
    
    async def _get_column_constraints(self, schema: str, table: str, column: str) -> Dict[str, Any]:
        """Get constraint information for a specific column"""
        try:
            constraints_info = {
                "constraints": [],
                "is_primary_key": False,
                "is_foreign_key": False,
                "foreign_key_reference": None
            }
            
            # Check for primary key
            self.cursor.execute("""
                SELECT tc.CONSTRAINT_NAME
                FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE ccu 
                    ON tc.CONSTRAINT_NAME = ccu.CONSTRAINT_NAME
                WHERE tc.TABLE_SCHEMA = ? 
                    AND tc.TABLE_NAME = ? 
                    AND ccu.COLUMN_NAME = ?
                    AND tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
            """, schema, table, column)
            
            if self.cursor.fetchone():
                constraints_info["is_primary_key"] = True
                constraints_info["constraints"].append("PRIMARY KEY")
            
            # Check for foreign keys
            self.cursor.execute("""
                SELECT 
                    fk.CONSTRAINT_NAME,
                    pk_table.TABLE_SCHEMA as REFERENCED_SCHEMA,
                    pk_table.TABLE_NAME as REFERENCED_TABLE,
                    pk_column.COLUMN_NAME as REFERENCED_COLUMN
                FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE fk 
                    ON rc.CONSTRAINT_NAME = fk.CONSTRAINT_NAME
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE pk_table 
                    ON rc.UNIQUE_CONSTRAINT_NAME = pk_table.CONSTRAINT_NAME
                JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE pk_column 
                    ON rc.UNIQUE_CONSTRAINT_NAME = pk_column.CONSTRAINT_NAME
                WHERE fk.TABLE_SCHEMA = ? 
                    AND fk.TABLE_NAME = ? 
                    AND fk.COLUMN_NAME = ?
            """, schema, table, column)
            
            fk_result = self.cursor.fetchone()
            if fk_result:
                constraints_info["is_foreign_key"] = True
                constraints_info["foreign_key_reference"] = f"{fk_result.REFERENCED_SCHEMA}.{fk_result.REFERENCED_TABLE}.{fk_result.REFERENCED_COLUMN}"
                constraints_info["constraints"].append(f"FOREIGN KEY REFERENCES {fk_result.REFERENCED_SCHEMA}.{fk_result.REFERENCED_TABLE}({fk_result.REFERENCED_COLUMN})")
            
            # Check for unique constraints
            self.cursor.execute("""
                SELECT tc.CONSTRAINT_NAME
                FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE ccu 
                    ON tc.CONSTRAINT_NAME = ccu.CONSTRAINT_NAME
                WHERE tc.TABLE_SCHEMA = ? 
                    AND tc.TABLE_NAME = ? 
                    AND ccu.COLUMN_NAME = ?
                    AND tc.CONSTRAINT_TYPE = 'UNIQUE'
            """, schema, table, column)
            
            if self.cursor.fetchone():
                constraints_info["constraints"].append("UNIQUE")
            
            # Check for check constraints
            self.cursor.execute("""
                SELECT cc.CHECK_CLAUSE
                FROM INFORMATION_SCHEMA.CHECK_CONSTRAINTS cc
                JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE ccu 
                    ON cc.CONSTRAINT_NAME = ccu.CONSTRAINT_NAME
                WHERE ccu.TABLE_SCHEMA = ? 
                    AND ccu.TABLE_NAME = ? 
                    AND ccu.COLUMN_NAME = ?
            """, schema, table, column)
            
            check_constraints = self.cursor.fetchall()
            for check in check_constraints:
                constraints_info["constraints"].append(f"CHECK {check.CHECK_CLAUSE}")
            
            return constraints_info
            
        except Exception as e:
            logger.warning(f"Failed to get constraints for {schema}.{table}.{column}: {e}")
            return {"constraints": [], "is_primary_key": False, "is_foreign_key": False, "foreign_key_reference": None}
    
    async def _get_sample_rows(self, schema: str, table: str, limit: int) -> List[Dict[str, Any]]:
        """Get sample rows with proper data type handling"""
        try:
            # Get column metadata first
            self.cursor.execute("""
                SELECT COLUMN_NAME, DATA_TYPE 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? 
                ORDER BY ORDINAL_POSITION
            """, schema, table)
            
            cols_meta = self.cursor.fetchall()
            
            # Build SELECT clause with proper casting
            select_parts = []
            for col_name, data_type in cols_meta:
                dt = data_type.lower()
                if dt in ("geography", "geometry", "hierarchyid", "sql_variant"):
                    select_parts.append(f"TRY_CONVERT(NVARCHAR(MAX), [{col_name}]) AS [{col_name}]")
                else:
                    select_parts.append(f"[{col_name}]")
            
            select_clause = ", ".join(select_parts)
            
            # Execute sample query
            query = f"SELECT TOP {limit} {select_clause} FROM [{schema}].[{table}]"
            self.cursor.execute(query)
            
            cols = [c[0] for c in self.cursor.description]
            rows = self.cursor.fetchall()
            
            # Convert to list of dictionaries
            result = []
            for row in rows:
                result.append({cols[i]: row[i] for i in range(len(cols))})
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to get sample rows for {schema}.{table}: {e}")
            return []
    
    async def _get_table_statistics(self, schema: str, table: str) -> Dict[str, Any]:
        """Get table statistics"""
        try:
            stats = {}
            
            # Get row count
            try:
                self.cursor.execute(f"SELECT COUNT(*) as row_count FROM [{schema}].[{table}]")
                result = self.cursor.fetchone()
                stats["row_count"] = result.row_count if result else 0
            except:
                stats["row_count"] = "unknown"
            
            # Get table size information
            try:
                self.cursor.execute("""
                    SELECT 
                        SUM(a.total_pages) * 8 AS TotalSpaceKB,
                        SUM(a.used_pages) * 8 AS UsedSpaceKB,
                        (SUM(a.total_pages) - SUM(a.used_pages)) * 8 AS UnusedSpaceKB
                    FROM sys.tables t
                    INNER JOIN sys.indexes i ON t.object_id = i.object_id
                    INNER JOIN sys.partitions p ON i.object_id = p.object_id AND i.index_id = p.index_id
                    INNER JOIN sys.allocation_units a ON p.partition_id = a.container_id
                    INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
                    WHERE s.name = ? AND t.name = ?
                    GROUP BY t.name, s.name
                """, schema, table)
                
                result = self.cursor.fetchone()
                if result:
                    stats["total_size_kb"] = result.TotalSpaceKB
                    stats["used_size_kb"] = result.UsedSpaceKB
                    stats["unused_size_kb"] = result.UnusedSpaceKB
            except:
                pass
            
            return stats
            
        except Exception as e:
            logger.warning(f"Failed to get statistics for {schema}.{table}: {e}")
            return {}
    
    async def _get_table_constraints(self, schema: str, table: str) -> Dict[str, Any]:
        """Get all constraints for a table"""
        try:
            constraints = {
                "primary_keys": [],
                "foreign_keys": [],
                "unique_constraints": [],
                "check_constraints": []
            }
            
            # Primary keys
            self.cursor.execute("""
                SELECT ccu.COLUMN_NAME
                FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE ccu 
                    ON tc.CONSTRAINT_NAME = ccu.CONSTRAINT_NAME
                WHERE tc.TABLE_SCHEMA = ? 
                    AND tc.TABLE_NAME = ? 
                    AND tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
            """, schema, table)
            
            constraints["primary_keys"] = [row.COLUMN_NAME for row in self.cursor.fetchall()]
            
            # Foreign keys
            self.cursor.execute("""
                SELECT 
                    fk.COLUMN_NAME,
                    pk_table.TABLE_SCHEMA as REFERENCED_SCHEMA,
                    pk_table.TABLE_NAME as REFERENCED_TABLE,
                    pk_column.COLUMN_NAME as REFERENCED_COLUMN
                FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE fk 
                    ON rc.CONSTRAINT_NAME = fk.CONSTRAINT_NAME
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE pk_table 
                    ON rc.UNIQUE_CONSTRAINT_NAME = pk_table.CONSTRAINT_NAME
                JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE pk_column 
                    ON rc.UNIQUE_CONSTRAINT_NAME = pk_column.CONSTRAINT_NAME
                WHERE fk.TABLE_SCHEMA = ? AND fk.TABLE_NAME = ?
            """, schema, table)
            
            for row in self.cursor.fetchall():
                constraints["foreign_keys"].append({
                    "column": row.COLUMN_NAME,
                    "references": f"{row.REFERENCED_SCHEMA}.{row.REFERENCED_TABLE}.{row.REFERENCED_COLUMN}"
                })
            
            # Add dbt constraints if available
            if self.dbt_extractor:
                dbt_constraints = await self.dbt_extractor.get_constraint_info(schema, table)
                constraints["dbt_business_rules"] = dbt_constraints.get("business_rules", [])
                constraints["dbt_tests"] = {
                    "not_null": dbt_constraints.get("not_null_constraints", []),
                    "unique": dbt_constraints.get("unique_constraints", []),
                    "check": dbt_constraints.get("check_constraints", [])
                }
            
            return constraints
            
        except Exception as e:
            logger.error(f"Failed to get constraints for {schema}.{table}: {e}")
            return {}
    
    async def _get_all_relationships(self) -> Dict[str, List[Dict[str, str]]]:
        """Get all foreign key relationships in the database"""
        try:
            self.cursor.execute("""
                SELECT 
                    fk_schema.name AS FK_SCHEMA,
                    fk_table.name AS FK_TABLE,
                    fk_column.name AS FK_COLUMN,
                    pk_schema.name AS PK_SCHEMA,
                    pk_table.name AS PK_TABLE,
                    pk_column.name AS PK_COLUMN,
                    fk.name AS FK_NAME
                FROM sys.foreign_keys fk
                JOIN sys.tables fk_table ON fk.parent_object_id = fk_table.object_id
                JOIN sys.schemas fk_schema ON fk_table.schema_id = fk_schema.schema_id
                JOIN sys.tables pk_table ON fk.referenced_object_id = pk_table.object_id
                JOIN sys.schemas pk_schema ON pk_table.schema_id = pk_schema.schema_id
                JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
                JOIN sys.columns fk_column ON fkc.parent_object_id = fk_column.object_id 
                    AND fkc.parent_column_id = fk_column.column_id
                JOIN sys.columns pk_column ON fkc.referenced_object_id = pk_column.object_id 
                    AND fkc.referenced_column_id = pk_column.column_id
                ORDER BY fk_schema.name, fk_table.name, fk_column.name
            """)
            
            relationships = {}
            for row in self.cursor.fetchall():
                fk_table = f"{row.FK_SCHEMA}.{row.FK_TABLE}"
                
                if fk_table not in relationships:
                    relationships[fk_table] = []
                
                relationships[fk_table].append({
                    "foreign_key_column": row.FK_COLUMN,
                    "references_schema": row.PK_SCHEMA,
                    "references_table": row.PK_TABLE,
                    "references_column": row.PK_COLUMN,
                    "constraint_name": row.FK_NAME
                })
            
            return relationships
            
        except Exception as e:
            logger.error(f"Failed to get relationships: {e}")
            return {}
    
    async def close(self):
        """Close database connection"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            logger.info("SQL connection closed")
        except Exception as e:
            logger.error(f"Error closing SQL connection: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()