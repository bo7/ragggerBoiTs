"""
dbt Integration Module
=====================

Extract semantic information from dbt models, docs, and lineage
to enhance RAG embeddings with business context.
"""

import json
import logging
import os
import subprocess
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ColumnInfo:
    """Column information with business context"""
    name: str
    data_type: str
    description: Optional[str] = None
    constraints: List[str] = None
    tests: List[str] = None
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_reference: Optional[str] = None


@dataclass
class TableInfo:
    """Table information with dbt context"""
    schema: str
    name: str
    model_type: str  # 'source', 'model', 'seed', etc.
    description: Optional[str] = None
    columns: List[ColumnInfo] = None
    constraints: List[str] = None
    tests: List[str] = None
    dependencies: List[str] = None
    business_rules: List[str] = None


class DBTSemanticExtractor:
    """Extract semantic information from dbt project"""
    
    def __init__(self, 
                 dbt_project_dir: Path = None,
                 profiles_dir: Path = None):
        self.dbt_project_dir = dbt_project_dir or Path.cwd()
        self.profiles_dir = profiles_dir or Path.home() / '.dbt'
        
        # dbt artifacts
        self.target_dir = self.dbt_project_dir / 'target'
        self.manifest_path = self.target_dir / 'manifest.json'
        self.catalog_path = self.target_dir / 'catalog.json'
        self.run_results_path = self.target_dir / 'run_results.json'
        
        # Parsed data
        self.manifest = {}
        self.catalog = {}
        self.sources = {}
        self.models = {}
        
    async def initialize(self) -> bool:
        """Initialize dbt integration and parse artifacts"""
        try:
            # Ensure dbt artifacts exist
            if not await self._ensure_dbt_artifacts():
                logger.error("Failed to generate dbt artifacts")
                return False
            
            # Parse dbt artifacts
            await self._parse_manifest()
            await self._parse_catalog()
            await self._parse_sources()
            
            logger.info("dbt semantic extractor initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize dbt extractor: {e}")
            return False
    
    async def _ensure_dbt_artifacts(self) -> bool:
        """Ensure dbt artifacts are generated and up-to-date"""
        try:
            logger.info("Generating dbt artifacts...")
            
            # Change to dbt project directory
            original_cwd = os.getcwd()
            os.chdir(self.dbt_project_dir)
            
            try:
                # Run dbt parse to generate manifest
                result = subprocess.run(
                    ['dbt', 'parse'],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    logger.error(f"dbt parse failed: {result.stderr}")
                    return False
                
                # Run dbt docs generate to create catalog
                result = subprocess.run(
                    ['dbt', 'docs', 'generate'],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode != 0:
                    logger.warning(f"dbt docs generate had issues: {result.stderr}")
                    # Continue anyway as manifest might be sufficient
                
                return True
                
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            logger.error(f"Failed to generate dbt artifacts: {e}")
            return False
    
    async def _parse_manifest(self):
        """Parse dbt manifest.json for model definitions and lineage"""
        try:
            if not self.manifest_path.exists():
                logger.warning("manifest.json not found")
                return
            
            with open(self.manifest_path, 'r') as f:
                self.manifest = json.load(f)
            
            logger.info(f"Parsed dbt manifest with {len(self.manifest.get('nodes', {}))} nodes")
            
        except Exception as e:
            logger.error(f"Failed to parse manifest: {e}")
    
    async def _parse_catalog(self):
        """Parse dbt catalog.json for column statistics and metadata"""
        try:
            if not self.catalog_path.exists():
                logger.warning("catalog.json not found - running without column stats")
                return
            
            with open(self.catalog_path, 'r') as f:
                self.catalog = json.load(f)
            
            logger.info(f"Parsed dbt catalog with {len(self.catalog.get('nodes', {}))} nodes")
            
        except Exception as e:
            logger.error(f"Failed to parse catalog: {e}")
    
    async def _parse_sources(self):
        """Parse sources.yml files for source definitions"""
        try:
            # Find all sources.yml files
            sources_files = list(self.dbt_project_dir.rglob('sources.yml'))
            sources_files.extend(list(self.dbt_project_dir.rglob('*sources*.yml')))
            
            for sources_file in sources_files:
                try:
                    with open(sources_file, 'r') as f:
                        sources_data = yaml.safe_load(f)
                    
                    if 'sources' in sources_data:
                        for source in sources_data['sources']:
                            source_name = source.get('name')
                            if source_name:
                                self.sources[source_name] = source
                                
                except Exception as e:
                    logger.warning(f"Failed to parse {sources_file}: {e}")
            
            logger.info(f"Parsed {len(self.sources)} dbt sources")
            
        except Exception as e:
            logger.error(f"Failed to parse sources: {e}")
    
    async def get_table_semantic_info(self, schema: str, table: str) -> TableInfo:
        """Get comprehensive semantic information for a table"""
        try:
            # Normalize names
            schema_lower = schema.lower()
            table_lower = table.lower()
            
            # Check if it's a source table
            source_info = self._get_source_info(schema_lower, table_lower)
            if source_info:
                return source_info
            
            # Check if it's a dbt model
            model_info = self._get_model_info(schema_lower, table_lower)
            if model_info:
                return model_info
            
            # Fallback - create basic info
            return TableInfo(
                schema=schema,
                name=table,
                model_type='unknown',
                columns=[],
                constraints=[],
                dependencies=[]
            )
            
        except Exception as e:
            logger.error(f"Failed to get semantic info for {schema}.{table}: {e}")
            return TableInfo(schema=schema, name=table, model_type='error', columns=[])
    
    def _get_source_info(self, schema: str, table: str) -> Optional[TableInfo]:
        """Get information for source tables"""
        try:
            # Find source definition
            for source_name, source_data in self.sources.items():
                if source_data.get('schema', '').lower() == schema:
                    tables = source_data.get('tables', [])
                    for tbl in tables:
                        if tbl.get('name', '').lower() == table:
                            return self._build_table_info_from_source(source_data, tbl, schema, table)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get source info for {schema}.{table}: {e}")
            return None
    
    def _get_model_info(self, schema: str, table: str) -> Optional[TableInfo]:
        """Get information for dbt models"""
        try:
            # Search manifest for model
            nodes = self.manifest.get('nodes', {})
            
            for node_id, node_data in nodes.items():
                if (node_data.get('schema', '').lower() == schema and 
                    node_data.get('name', '').lower() == table):
                    return self._build_table_info_from_model(node_data, schema, table)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get model info for {schema}.{table}: {e}")
            return None
    
    def _build_table_info_from_source(self, source_data: Dict, table_data: Dict, schema: str, table: str) -> TableInfo:
        """Build TableInfo from source definition"""
        try:
            columns = []
            table_columns = table_data.get('columns', [])
            
            for col in table_columns:
                col_info = ColumnInfo(
                    name=col.get('name', ''),
                    data_type=col.get('data_type', 'unknown'),
                    description=col.get('description'),
                    tests=col.get('tests', []),
                    constraints=[]
                )
                
                # Extract constraints from tests
                col_info.constraints = self._extract_constraints_from_tests(col.get('tests', []))
                
                columns.append(col_info)
            
            # Get table-level info
            table_info = TableInfo(
                schema=schema,
                name=table,
                model_type='source',
                description=table_data.get('description'),
                columns=columns,
                tests=table_data.get('tests', []),
                constraints=self._extract_constraints_from_tests(table_data.get('tests', [])),
                dependencies=[]
            )
            
            return table_info
            
        except Exception as e:
            logger.error(f"Failed to build source table info: {e}")
            return TableInfo(schema=schema, name=table, model_type='source_error', columns=[])
    
    def _build_table_info_from_model(self, node_data: Dict, schema: str, table: str) -> TableInfo:
        """Build TableInfo from dbt model"""
        try:
            columns = []
            
            # Get column info from manifest
            node_columns = node_data.get('columns', {})
            for col_name, col_data in node_columns.items():
                col_info = ColumnInfo(
                    name=col_name,
                    data_type=col_data.get('data_type', 'unknown'),
                    description=col_data.get('description'),
                    tests=list(col_data.get('tests', [])),
                    constraints=self._extract_constraints_from_tests(col_data.get('tests', []))
                )
                columns.append(col_info)
            
            # Get catalog info if available
            self._enhance_columns_with_catalog(columns, schema, table)
            
            # Get dependencies
            dependencies = node_data.get('depends_on', {}).get('nodes', [])
            
            table_info = TableInfo(
                schema=schema,
                name=table,
                model_type=node_data.get('resource_type', 'model'),
                description=node_data.get('description'),
                columns=columns,
                tests=list(node_data.get('tests', [])),
                constraints=self._extract_constraints_from_tests(node_data.get('tests', [])),
                dependencies=dependencies,
                business_rules=self._extract_business_rules(node_data)
            )
            
            return table_info
            
        except Exception as e:
            logger.error(f"Failed to build model table info: {e}")
            return TableInfo(schema=schema, name=table, model_type='model_error', columns=[])
    
    def _enhance_columns_with_catalog(self, columns: List[ColumnInfo], schema: str, table: str):
        """Enhance column info with catalog statistics"""
        try:
            catalog_nodes = self.catalog.get('nodes', {})
            
            # Find matching catalog entry
            for node_id, node_data in catalog_nodes.items():
                if (node_data.get('metadata', {}).get('schema', '').lower() == schema.lower() and
                    node_data.get('metadata', {}).get('name', '').lower() == table.lower()):
                    
                    catalog_columns = node_data.get('columns', {})
                    
                    for col in columns:
                        if col.name.lower() in catalog_columns:
                            cat_col = catalog_columns[col.name.lower()]
                            
                            # Update data type if more specific
                            if cat_col.get('type') and col.data_type == 'unknown':
                                col.data_type = cat_col['type']
                            
                            # Add any additional metadata
                            if cat_col.get('comment'):
                                col.description = col.description or cat_col['comment']
                    
                    break
                    
        except Exception as e:
            logger.warning(f"Failed to enhance columns with catalog: {e}")
    
    def _extract_constraints_from_tests(self, tests: List) -> List[str]:
        """Extract database constraints from dbt tests"""
        constraints = []
        
        for test in tests:
            if isinstance(test, str):
                if test in ['not_null', 'unique']:
                    constraints.append(test.upper())
            elif isinstance(test, dict):
                for test_name, test_config in test.items():
                    if test_name == 'not_null':
                        constraints.append('NOT NULL')
                    elif test_name == 'unique':
                        constraints.append('UNIQUE')
                    elif test_name == 'accepted_values':
                        values = test_config.get('values', [])
                        if values:
                            constraints.append(f"CHECK IN ({', '.join(map(str, values))})")
                    elif test_name == 'relationships':
                        ref_table = test_config.get('to', '')
                        ref_field = test_config.get('field', '')
                        if ref_table and ref_field:
                            constraints.append(f"FOREIGN KEY REFERENCES {ref_table}({ref_field})")
        
        return constraints
    
    def _extract_business_rules(self, node_data: Dict) -> List[str]:
        """Extract business rules from model definition"""
        rules = []
        
        # Extract from model SQL
        raw_sql = node_data.get('raw_sql', '')
        if 'where' in raw_sql.lower():
            rules.append("Has filtering logic in WHERE clause")
        
        if 'case when' in raw_sql.lower():
            rules.append("Contains conditional business logic")
        
        if 'join' in raw_sql.lower():
            rules.append("Combines data from multiple sources")
        
        # Extract from model config
        config = node_data.get('config', {})
        if config.get('materialized') == 'incremental':
            rules.append("Incremental model - processes only new/changed data")
        
        return rules
    
    async def get_lineage_info(self, schema: str, table: str) -> Dict[str, List[str]]:
        """Get data lineage for a table"""
        try:
            lineage = {
                'upstream': [],
                'downstream': []
            }
            
            # Find all nodes that depend on this table
            nodes = self.manifest.get('nodes', {})
            
            target_identifier = f"{schema.lower()}.{table.lower()}"
            
            for node_id, node_data in nodes.items():
                dependencies = node_data.get('depends_on', {}).get('nodes', [])
                
                # Check if this node depends on our target table
                for dep in dependencies:
                    if target_identifier in dep.lower():
                        lineage['downstream'].append(node_id)
                
                # Check if our target table depends on this node
                node_schema = node_data.get('schema', '').lower()
                node_name = node_data.get('name', '').lower()
                
                if f"{node_schema}.{node_name}" == target_identifier:
                    lineage['upstream'] = dependencies
            
            return lineage
            
        except Exception as e:
            logger.error(f"Failed to get lineage for {schema}.{table}: {e}")
            return {'upstream': [], 'downstream': []}
    
    async def generate_semantic_description(self, table_info: TableInfo) -> str:
        """Generate rich semantic description for embeddings"""
        try:
            parts = []
            
            # Table description
            if table_info.description:
                parts.append(f"Table Description: {table_info.description}")
            
            parts.append(f"Schema: {table_info.schema}")
            parts.append(f"Table Type: {table_info.model_type}")
            
            # Column information
            if table_info.columns:
                parts.append(f"Contains {len(table_info.columns)} columns:")
                
                for col in table_info.columns:
                    col_desc = f"- {col.name} ({col.data_type})"
                    
                    if col.description:
                        col_desc += f": {col.description}"
                    
                    if col.constraints:
                        col_desc += f" [Constraints: {', '.join(col.constraints)}]"
                    
                    parts.append(col_desc)
            
            # Business rules
            if table_info.business_rules:
                parts.append("Business Rules:")
                for rule in table_info.business_rules:
                    parts.append(f"- {rule}")
            
            # Dependencies
            if table_info.dependencies:
                parts.append(f"Depends on: {', '.join(table_info.dependencies)}")
            
            return "\n".join(parts)
            
        except Exception as e:
            logger.error(f"Failed to generate semantic description: {e}")
            return f"Table: {table_info.schema}.{table_info.name}"
    
    async def get_all_semantic_info(self) -> Dict[str, TableInfo]:
        """Get semantic information for all tables in the project"""
        try:
            all_tables = {}
            
            # Process sources
            for source_name, source_data in self.sources.items():
                schema = source_data.get('schema', source_name)
                tables = source_data.get('tables', [])
                
                for table_data in tables:
                    table_name = table_data.get('name')
                    if table_name:
                        key = f"{schema}.{table_name}"
                        all_tables[key] = await self.get_table_semantic_info(schema, table_name)
            
            # Process models
            nodes = self.manifest.get('nodes', {})
            for node_id, node_data in nodes.items():
                if node_data.get('resource_type') in ['model', 'seed']:
                    schema = node_data.get('schema')
                    table_name = node_data.get('name')
                    
                    if schema and table_name:
                        key = f"{schema}.{table_name}"
                        all_tables[key] = await self.get_table_semantic_info(schema, table_name)
            
            logger.info(f"Extracted semantic info for {len(all_tables)} tables")
            return all_tables
            
        except Exception as e:
            logger.error(f"Failed to get all semantic info: {e}")
            return {}
    
    async def get_constraint_info(self, schema: str, table: str) -> Dict[str, Any]:
        """Get detailed constraint information for RAG system"""
        try:
            table_info = await self.get_table_semantic_info(schema, table)
            
            constraints_info = {
                'primary_keys': [],
                'foreign_keys': [],
                'unique_constraints': [],
                'check_constraints': [],
                'not_null_constraints': [],
                'business_rules': table_info.business_rules or []
            }
            
            for col in table_info.columns:
                for constraint in col.constraints:
                    if 'PRIMARY KEY' in constraint:
                        constraints_info['primary_keys'].append(col.name)
                    elif 'FOREIGN KEY' in constraint:
                        constraints_info['foreign_keys'].append({
                            'column': col.name,
                            'reference': constraint
                        })
                    elif 'UNIQUE' in constraint:
                        constraints_info['unique_constraints'].append(col.name)
                    elif 'NOT NULL' in constraint:
                        constraints_info['not_null_constraints'].append(col.name)
                    elif 'CHECK' in constraint:
                        constraints_info['check_constraints'].append({
                            'column': col.name,
                            'constraint': constraint
                        })
            
            return constraints_info
            
        except Exception as e:
            logger.error(f"Failed to get constraint info: {e}")
            return {}