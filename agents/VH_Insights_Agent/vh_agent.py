import asyncio
import requests
from dotenv import load_dotenv
from uagents_core.contrib.protocols.chat import (
    chat_protocol_spec,
    ChatMessage,
    ChatAcknowledgement,
    TextContent,
)
from uagents import Agent, Context, Protocol
from datetime import datetime, timezone
from uuid import uuid4

# ============================================================================
# MCP IMPORTS (COMMENTED OUT - MIGRATING TO DIRECT POSTGRES)
# ============================================================================
# import mcp
# from mcp.client.streamable_http import streamablehttp_client
# from contextlib import AsyncExitStack

# ============================================================================
# DIRECT POSTGRES IMPORTS
# ============================================================================
import asyncpg
import json
# import base64  # Still needed for other purposes
import asyncio
from typing import Dict, Any, List
import os
from urllib.parse import quote_plus

# Load environment variables
load_dotenv()

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================
# MCP Environment Variables (keeping for potential fallback)
SUPABASE_ACCESS_TOKEN = os.getenv("SUPABASE_ACCESS_TOKEN")
SMITHERY_API_KEY = os.getenv("SMITHERY_API_KEY")
SUPABASE_PROJECT_ID = os.getenv("SUPABASE_PROJECT_ID")
ASI1_API_KEY = os.getenv("ASI1_API_KEY")

# Direct Postgres Environment Variables (NEW)
SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
# URL encode the password to handle special characters (like in data loaders)
SUPABASE_DB_PASSWORD = quote_plus(SUPABASE_DB_PASSWORD) if SUPABASE_DB_PASSWORD else ""
USE_DIRECT_DB = os.getenv("USE_DIRECT_DB", "true").lower() == "true"

# Check required environment variables based on connection type
if USE_DIRECT_DB:
    if not SUPABASE_DB_PASSWORD or not ASI1_API_KEY:
        raise ValueError("Missing required environment variables for direct DB: SUPABASE_DB_PASSWORD, ASI1_API_KEY")
else:
    if not SUPABASE_ACCESS_TOKEN or not SMITHERY_API_KEY or not SUPABASE_PROJECT_ID or not ASI1_API_KEY:
        raise ValueError("Missing required environment variables for MCP: SUPABASE_ACCESS_TOKEN, SMITHERY_API_KEY, SUPABASE_PROJECT_ID, ASI1_API_KEY")

# ============================================================================
# MCP CLIENT (COMMENTED OUT - MIGRATING TO DIRECT POSTGRES)
# ============================================================================
# class SupabaseMCPClient:
#     def __init__(self):
#         self.session = None
#         self.exit_stack = AsyncExitStack()
#         self.config = {
#             "accessToken": SUPABASE_ACCESS_TOKEN,
#             "readOnly": True,
#         }
#         self.project_id = SUPABASE_PROJECT_ID
# 
#     async def connect(self, ctx: Context):
#         config_b64 = base64.b64encode(json.dumps(self.config).encode())
#         url = f"https://server.smithery.ai/@supabase-community/supabase-mcp/mcp?config={config_b64}&api_key={SMITHERY_API_KEY}&profile=dual-barnacle-C2qHG5"
#         read_stream, write_stream, _ = await self.exit_stack.enter_async_context(
#             streamablehttp_client(url)
#         )
#         self.session = await self.exit_stack.enter_async_context(
#             mcp.ClientSession(read_stream, write_stream)
#         )
#         await self.session.initialize()
#         ctx.logger.info("Connected to Supabase MCP server")
# 
#     async def ensure_connection(self, ctx: Context):
#         if not self.session:
#             await self.connect(ctx)
#             return
#         
#         try:
#             await self.session.list_tools()
#         except Exception as e:
#             ctx.logger.warning(f"Session check failed: {str(e)}. Attempting to reconnect...")
#             await self.cleanup()
#             await self.connect(ctx)
# 
#     async def call_tool(self, tool_name: str, arguments: dict, ctx: Context):
#         await self.ensure_connection(ctx)
#         max_retries = 3
#         for attempt in range(max_retries):
#             try:
#                 return await self.session.call_tool(tool_name, arguments=arguments)
#             except Exception as e:
#                 if attempt == max_retries - 1:
#                     raise
#                 ctx.logger.warning(f"Tool call attempt {attempt + 1} failed: {str(e)}. Retrying...")
#                 await asyncio.sleep(1)
#                 await self.ensure_connection(ctx)
# 
#     async def cleanup(self):
#         await self.exit_stack.aclose()
#         self.session = None

# ============================================================================
# DIRECT POSTGRES CLIENT (NEW)
# ============================================================================
class SupabaseDirectClient:
    """Direct PostgreSQL client using asyncpg - replacement for MCP client"""
    
    def __init__(self):
        self.pool = None
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
        # Connection string using IP address instead of hostname to bypass DNS issues
        # IP addresses from nslookup: 3.139.14.59, 3.13.175.194
        self.dsn = f"postgresql://postgres.ylebxbxshnhtltukbjzx:{SUPABASE_DB_PASSWORD}@3.139.14.59:5432/postgres?sslmode=require"
    
    async def connect(self, ctx: Context):
        """Initialize connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                dsn=self.dsn,
                min_size=1,
                max_size=10,
                command_timeout=60
            )
            ctx.logger.info("✅ Connected to Supabase Postgres directly via asyncpg")
        except Exception as e:
            ctx.logger.error(f"❌ Failed to connect to Postgres: {str(e)}")
            raise
    
    async def ensure_connection(self, ctx: Context):
        """Ensure we have an active connection pool"""
        if not self.pool:
            await self.connect(ctx)
            return
        
        try:
            # Test connection with simple query
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
        except Exception as e:
            ctx.logger.warning(f"Connection check failed: {str(e)}. Attempting to reconnect...")
            await self.cleanup()
            await self.connect(ctx)
    
    async def call_tool(self, tool_name: str, arguments: dict, ctx: Context):
        """
        Maintain same interface as MCP client for minimal code changes
        Currently only supports 'execute_sql' tool
        """
        if tool_name != "execute_sql":
            raise ValueError(f"Unsupported tool: {tool_name}")
        
        query = arguments.get("query")
        if not query:
            raise ValueError("Missing 'query' in arguments")
        
        return await self.execute_sql(query, ctx)
    
    async def execute_sql(self, query: str, ctx: Context):
        """Execute SQL query and return results in MCP-compatible format"""
        await self.ensure_connection(ctx)
        
        for attempt in range(self.max_retries):
            try:
                async with self.pool.acquire() as conn:
                    # Execute query and fetch all results
                    rows = await conn.fetch(query)
                    
                    # Convert asyncpg Records to list of dicts (same format as MCP)
                    data = [dict(row) for row in rows]
                    
                    # Return in MCP-compatible format
                    class MockResult:
                        def __init__(self, data):
                            self.content = [MockContent(json.dumps(data))]
                    
                    class MockContent:
                        def __init__(self, text):
                            self.text = text
                    
                    return MockResult(data)
                    
            except Exception as e:
                if attempt == self.max_retries - 1:
                    ctx.logger.error(f"SQL execution failed after {self.max_retries} attempts: {str(e)}")
                    raise
                ctx.logger.warning(f"SQL attempt {attempt + 1} failed: {str(e)}. Retrying...")
                await asyncio.sleep(self.retry_delay)
                await self.ensure_connection(ctx)
    
    async def cleanup(self):
        """Cleanup connection pool"""
        if self.pool:
            await self.pool.close()
            self.pool = None

# ASI1 configuration
ASI1_URL = "https://api.asi1.ai/v1/chat/completions"
ASI1_HEADERS = {
    "Authorization": f"Bearer {ASI1_API_KEY}",
    "Content-Type": "application/json"
}

class VaccineHesitancyAgent:
    def __init__(self, mcp_client: SupabaseDirectClient):
        self.mcp_client = mcp_client
        
    async def discover_database_schema(self, ctx: Context) -> Dict[str, Any]:
        """Query the database to discover actual schema and valid values for ALL 18 schemas"""
        
        # Initialize schema info for all 18 vaccine hesitancy schemas
        schema_info = {
            'valid_measures': [],
            'valid_age_groups': [],
            'valid_regions': [],
            'valid_sex_values': [],
            'valid_ethnicities': [],
            'valid_religions': [],
            'valid_disability_status': [],
            'valid_cev_status': [],
            'valid_health_conditions': [],
            'valid_health_general_conditions': [],
            'valid_imd_quintiles': [],
            'valid_employment_status': [],
            'valid_expense_affordability': [],
            'valid_household_types': [],
            'valid_caregiver_status': [],
            'valid_age_sex_groups': [],
            'valid_trends_periods': [],
            'valid_barriers_groups': [],
            'discovery_time': datetime.now(timezone.utc).isoformat()
        }
        
        try:
            # Discover valid measure values from hesitancy_reasons
            measure_query = "SELECT DISTINCT measure FROM vaccine_hesitancy_reasons WHERE measure IS NOT NULL AND measure != '' LIMIT 20;"
            try:
                result = await self.mcp_client.call_tool("execute_sql", {
                    "project_id": SUPABASE_PROJECT_ID,
                    "query": measure_query
                }, ctx)
                content = result.content[0].text if isinstance(result.content, list) else result.content
                data = json.loads(content)
                schema_info['valid_measures'] = [str(row.get('measure', '')) for row in data if row.get('measure')]
                ctx.logger.info(f"Discovered {len(schema_info['valid_measures'])} valid measures")
            except Exception as e:
                ctx.logger.warning(f"Failed to discover measures: {e}")
                schema_info['valid_measures'] = ['Unknown']
            
            # Discover valid age groups
            age_query = "SELECT DISTINCT age_group FROM vaccine_hesitancy_age_group WHERE age_group IS NOT NULL AND age_group != '' LIMIT 20;"
            try:
                result = await self.mcp_client.call_tool("execute_sql", {
                    "project_id": SUPABASE_PROJECT_ID,
                    "query": age_query
                }, ctx)
                content = result.content[0].text if isinstance(result.content, list) else result.content
                data = json.loads(content)
                schema_info['valid_age_groups'] = [str(row.get('age_group', '')) for row in data if row.get('age_group')]
                ctx.logger.info(f"Discovered {len(schema_info['valid_age_groups'])} valid age groups")
            except Exception as e:
                ctx.logger.warning(f"Failed to discover age groups: {e}")
                schema_info['valid_age_groups'] = ['Unknown']
            
            # Discover valid regions
            region_query = "SELECT DISTINCT region FROM vaccine_hesitancy_region WHERE region IS NOT NULL AND region != '' LIMIT 20;"
            try:
                result = await self.mcp_client.call_tool("execute_sql", {
                    "project_id": SUPABASE_PROJECT_ID,
                    "query": region_query
                }, ctx)
                content = result.content[0].text if isinstance(result.content, list) else result.content
                data = json.loads(content)
                schema_info['valid_regions'] = [str(row.get('region', '')) for row in data if row.get('region')]
                ctx.logger.info(f"Discovered {len(schema_info['valid_regions'])} valid regions")
            except Exception as e:
                ctx.logger.warning(f"Failed to discover regions: {e}")
                schema_info['valid_regions'] = ['Unknown']
            
            # Discover valid sex values
            sex_query = "SELECT DISTINCT sex FROM vaccine_hesitancy_sex WHERE sex IS NOT NULL AND sex != '' LIMIT 10;"
            try:
                result = await self.mcp_client.call_tool("execute_sql", {
                    "project_id": SUPABASE_PROJECT_ID,
                    "query": sex_query
                }, ctx)
                content = result.content[0].text if isinstance(result.content, list) else result.content
                data = json.loads(content)
                schema_info['valid_sex_values'] = [str(row.get('sex', '')) for row in data if row.get('sex')]
                ctx.logger.info(f"Discovered {len(schema_info['valid_sex_values'])} valid sex values")
            except Exception as e:
                ctx.logger.warning(f"Failed to discover sex values: {e}")
                schema_info['valid_sex_values'] = ['Unknown']
            
            # Discover valid ethnicities
            ethnicity_query = "SELECT DISTINCT ethnicity FROM vaccine_hesitancy_ethnicity WHERE ethnicity IS NOT NULL AND ethnicity != '' LIMIT 20;"
            try:
                result = await self.mcp_client.call_tool("execute_sql", {
                    "project_id": SUPABASE_PROJECT_ID,
                    "query": ethnicity_query
                }, ctx)
                content = result.content[0].text if isinstance(result.content, list) else result.content
                data = json.loads(content)
                schema_info['valid_ethnicities'] = [str(row.get('ethnicity', '')) for row in data if row.get('ethnicity')]
                ctx.logger.info(f"Discovered {len(schema_info['valid_ethnicities'])} valid ethnicities")
            except Exception as e:
                ctx.logger.warning(f"Failed to discover ethnicities: {e}")
                schema_info['valid_ethnicities'] = ['Unknown']
            
            # Discover additional schemas for comprehensive coverage
            additional_schemas = [
                ('valid_religions', 'vaccine_hesitancy_religion', 'religion'),
                ('valid_disability_status', 'vaccine_hesitancy_disability', 'disability_status'),
                ('valid_cev_status', 'vaccine_hesitancy_cev', 'cev_status'),
                ('valid_health_conditions', 'vaccine_hesitancy_health_condition', 'health_condition'),
                ('valid_health_general_conditions', 'vaccine_hesitancy_health_general_condition', 'health_general_condition'),
                ('valid_imd_quintiles', 'vaccine_hesitancy_imd_quintile', 'imd_quintile'),
                ('valid_employment_status', 'vaccine_hesitancy_employment', 'employment_status'),
                ('valid_expense_affordability', 'vaccine_hesitancy_expense_affordability', 'expense_affordability'),
                ('valid_household_types', 'vaccine_hesitancy_household_type', 'household_type'),
                ('valid_caregiver_status', 'vaccine_hesitancy_caregiver_status', 'caregiver_status'),
                ('valid_age_sex_groups', '"vaccine_hesitancy_age_sex_group"', '"group"'),  # Escape BOTH table and column names
                ('valid_trends_periods', 'vaccine_hesitancy_trends', 'period'),
                ('valid_barriers_groups', '"vaccine_hesitancy_barriers_group"', '"group"')  # Escape BOTH table and column names
            ]
            
            for schema_key, table_name, column_name in additional_schemas:
                # Special handling for reserved words already quoted in both table and column names
                if column_name.startswith('"') and column_name.endswith('"'):
                    # Both table and column names are already quoted (for reserved words)
                    query = f"SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL AND {column_name} != '' LIMIT 15;"
                else:
                    # Standard case - no reserved words
                    query = f"SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL AND {column_name} != '' LIMIT 15;"
                try:
                    result = await self.mcp_client.call_tool("execute_sql", {
                        "project_id": SUPABASE_PROJECT_ID,
                        "query": query
                    }, ctx)
                    content = result.content[0].text if isinstance(result.content, list) else result.content
                    data = json.loads(content)
                    schema_info[schema_key] = [str(row.get(column_name, '')) for row in data if row.get(column_name)]
                    ctx.logger.info(f"Discovered {len(schema_info[schema_key])} valid {schema_key.replace('valid_', '')}")
                except Exception as e:
                    ctx.logger.warning(f"Failed to discover {schema_key}: {e}")
                    schema_info[schema_key] = ['Unknown']
        except Exception as e:
            ctx.logger.error(f"Schema discovery failed: {e}")
            # Fallback to basic values for all schemas
            schema_info = {
                'valid_measures': ['Unknown'],
                'valid_age_groups': ['Unknown'],
                'valid_regions': ['Unknown'],
                'valid_sex_values': ['Unknown'],
                'valid_ethnicities': ['Unknown'],
                'valid_religions': ['Unknown'],
                'valid_disability_status': ['Unknown'],
                'valid_cev_status': ['Unknown'],
                'valid_health_conditions': ['Unknown'],
                'valid_health_general_conditions': ['Unknown'],
                'valid_imd_quintiles': ['Unknown'],
                'valid_employment_status': ['Unknown'],
                'valid_expense_affordability': ['Unknown'],
                'valid_household_types': ['Unknown'],
                'valid_caregiver_status': ['Unknown'],
                'valid_age_sex_groups': ['Unknown'],
                'valid_trends_periods': ['Unknown'],
                'valid_barriers_groups': ['Unknown'],
                'discovery_time': datetime.now(timezone.utc).isoformat()
            }
        
        return schema_info

    def generate_dynamic_ons_prompt(self, schema_info: Dict[str, Any]) -> str:
        """Generate dynamic ONS prompt based on discovered database schema for ALL 18 schemas"""
        
        # Get actual values from database (or fallback if discovery failed)
        valid_measures = schema_info.get('valid_measures', [])[:8]  # Limit to first 8 for space
        valid_age_groups = schema_info.get('valid_age_groups', [])[:8]
        valid_regions = schema_info.get('valid_regions', [])[:8]
        valid_sex_values = schema_info.get('valid_sex_values', [])[:3]
        valid_ethnicities = schema_info.get('valid_ethnicities', [])[:6]
        valid_religions = schema_info.get('valid_religions', [])[:6]
        valid_disability_status = schema_info.get('valid_disability_status', [])[:4]
        valid_cev_status = schema_info.get('valid_cev_status', [])[:3]
        valid_health_conditions = schema_info.get('valid_health_conditions', [])[:6]
        valid_imd_quintiles = schema_info.get('valid_imd_quintiles', [])[:5]
        valid_employment_status = schema_info.get('valid_employment_status', [])[:6]
        
        # Create dynamic examples based on actual data
        measure_examples = ', '.join([f"'{m}'" for m in valid_measures if m != 'Unknown']) if valid_measures and valid_measures != ['Unknown'] else "'Unknown'"
        age_examples = ', '.join([f"'{a}'" for a in valid_age_groups if a != 'Unknown']) if valid_age_groups and valid_age_groups != ['Unknown'] else "'Unknown'"
        region_examples = ', '.join([f"'{r}'" for r in valid_regions if r != 'Unknown']) if valid_regions and valid_regions != ['Unknown'] else "'Unknown'"
        
        return f"""You are an ONS vaccine hesitancy data specialist. Generate intelligent SQL queries for these tables:

**AVAILABLE ONS TABLES & COLUMNS:**

vaccine_hesitancy_region: region, measure, percent, weighted_count, sample_size
vaccine_hesitancy_age: wave_date, age_band, measure, percent, lcl, ucl, weighted_count, sample_size  
vaccine_hesitancy_age_group: wave_date, age_group, measure, percent, lcl, ucl, weighted_count, sample_size
vaccine_hesitancy_sex: wave_date, sex, measure, percent, lcl, ucl, weighted_count, sample_size
vaccine_hesitancy_ethnicity: wave_date, ethnicity, measure, percent, lcl, ucl, weighted_count, sample_size
vaccine_hesitancy_religion: wave_date, religion, measure, percent, lcl, ucl, weighted_count, sample_size
vaccine_hesitancy_disability: wave_date, disability_status, measure, percent, lcl, ucl, weighted_count, sample_size
vaccine_hesitancy_cev: wave_date, cev_status, measure, percent, lcl, ucl, weighted_count, sample_size
vaccine_hesitancy_health_condition: wave_date, health_condition, measure, percent, lcl, ucl, weighted_count, sample_size
vaccine_hesitancy_health_general_condition: wave_date, health_general_condition, measure, percent, lcl, ucl, weighted_count, sample_size
vaccine_hesitancy_imd_quintile: wave_date, imd_quintile, measure, percent, lcl, ucl, weighted_count, sample_size
vaccine_hesitancy_employment: wave_date, employment_status, measure, percent, lcl, ucl, weighted_count, sample_size
vaccine_hesitancy_expense_affordability: wave_date, expense_affordability, measure, percent, lcl, ucl, weighted_count, sample_size
vaccine_hesitancy_household_type: wave_date, household_type, measure, percent, lcl, ucl, weighted_count, sample_size
vaccine_hesitancy_caregiver_status: wave_date, caregiver_status, measure, percent, lcl, ucl, weighted_count, sample_size
vaccine_hesitancy_age_sex: wave_date, "group", subgroup, measure, value, weighted_count, sample_size
vaccine_hesitancy_trends: period, block, measure, value_type, value, weighted_count, sample_size
vaccine_hesitancy_barriers: block, "group", measure, value_type, value, weighted_count, sample_size
vaccine_hesitancy_reasons: period, "group", block, measure, percent, lcl, ucl, weighted_count, sample_size

**CRITICAL SQL RULES:**
- NEVER JOIN tables - query one table at a time (different schemas incompatible for JOINs)
- Use UNION if you need data from multiple tables
- vaccine_hesitancy_region has NO wave_date column (use measure, percent, region only)
- All other tables have wave_date column
- Always use single table queries: SELECT columns FROM single_table WHERE conditions

**ACTUAL VALID VALUES DISCOVERED IN DATABASE:**

**Valid Measure Values:** {', '.join(valid_measures) if valid_measures and valid_measures != ['Unknown'] else 'Use measure IS NOT NULL to find available values'}

**Valid Age Groups:** {', '.join(valid_age_groups) if valid_age_groups and valid_age_groups != ['Unknown'] else 'Use age_group IS NOT NULL to find available values'}

**Valid Regions:** {', '.join(valid_regions) if valid_regions and valid_regions != ['Unknown'] else 'Use region IS NOT NULL to find available values'}

**Valid Sex Values:** {', '.join(valid_sex_values) if valid_sex_values and valid_sex_values != ['Unknown'] else 'Use sex IS NOT NULL to find available values'}

**Valid Ethnicities:** {', '.join(valid_ethnicities) if valid_ethnicities and valid_ethnicities != ['Unknown'] else 'Use ethnicity IS NOT NULL to find available values'}

**Valid Religions:** {', '.join(valid_religions) if valid_religions and valid_religions != ['Unknown'] else 'Use religion IS NOT NULL to find available values'}

**Valid Disability Status:** {', '.join(valid_disability_status) if valid_disability_status and valid_disability_status != ['Unknown'] else 'Use disability_status IS NOT NULL to find available values'}

**Valid CEV Status:** {', '.join(valid_cev_status) if valid_cev_status and valid_cev_status != ['Unknown'] else 'Use cev_status IS NOT NULL to find available values'}

**Valid Health Conditions:** {', '.join(valid_health_conditions) if valid_health_conditions and valid_health_conditions != ['Unknown'] else 'Use health_condition IS NOT NULL to find available values'}

**Valid IMD Quintiles:** {', '.join(valid_imd_quintiles) if valid_imd_quintiles and valid_imd_quintiles != ['Unknown'] else 'Use imd_quintile IS NOT NULL to find available values'}

**Valid Employment Status:** {', '.join(valid_employment_status) if valid_employment_status and valid_employment_status != ['Unknown'] else 'Use employment_status IS NOT NULL to find available values'}

**PROGRESSIVE DATA DISCOVERY STRATEGY:**

**Primary Data Approach (Try First):**
- Query for specific hesitancy measures: "unlikely to have vaccine", "declined vaccine", "vaccine hesitancy"
- Include concern measures: "concerned about vaccine", "worried about side effects"
- Include uncertainty measures: "neither likely nor unlikely", "don't know", "prefer not to say"

**Fallback Data Approach (If Primary Returns <10 Records):**
- Expand to include broader vaccine measures: any measure containing "vaccine"
- Include acceptance measures for comparison: "received vaccine", "positive sentiment"
- Show complete data picture including N/A values

**Data Quality Strategy:**
- **First try**: WHERE percent IS NOT NULL (prioritize actual data)
- **If insufficient data**: Remove percent filter to show all available measures
- **Always show**: What data exists vs. what has actual values vs. what is N/A

**CRITICAL SQL RULES:**
- ALWAYS specify columns: SELECT measure, percent, age_group FROM table
- NEVER use: SELECT * FROM table (this causes errors)
- **Progressive filtering**: Start restrictive, then broaden if needed
- Use actual values from the lists above when possible
- If unsure about values, use IS NOT NULL instead of specific values
- **Avoid artificial limits**: Remove LIMIT clauses unless specifically needed for performance

**PROGRESSIVE QUERY STRATEGY:**

1. **For specific demographic queries (e.g., "Buddhist vaccine hesitancy"):**
   - **Step 1**: Try hesitancy-specific measures with percent IS NOT NULL
   - **Step 2**: If <5 records, expand to all vaccine measures
   - **Step 3**: If still <5 records, remove percent filter to show all available data
   - **Always show**: What data exists even if values are N/A

2. **For comparison queries (e.g., "age groups most hesitant"):**
   - **Step 1**: Filter for hesitancy measures with actual data
   - **Step 2**: If insufficient data, include broader measures
   - **Step 3**: Group by dimension and show complete picture

3. **For exploration queries (e.g., "vaccine data for region"):**
   - **Step 1**: Discover all available measures for the region
   - **Step 2**: Show both hesitancy and acceptance measures
   - **Step 3**: Include data quality information

4. **Data Quality Principles:**
   - **Never return 0 rows** when any data exists for the dimension
   - **Show data availability** even if incomplete
   - **Include N/A values** with clear labeling
   - **Provide context** about what data is missing vs. available

**PROGRESSIVE QUERY EXAMPLES:**

**Specific Demographic Query (Primary):** "Buddhist vaccine hesitancy"
→ `SELECT religion, measure, percent FROM vaccine_hesitancy_religion 
   WHERE religion = 'Buddhist'
   AND (measure LIKE '%unlikely%' OR measure LIKE '%declined%' OR measure LIKE '%hesitancy%')
   AND percent IS NOT NULL 
   ORDER BY percent DESC;`

**Specific Demographic Query (Fallback):** If primary returns <5 records:
→ `SELECT religion, measure, percent FROM vaccine_hesitancy_religion 
   WHERE religion = 'Buddhist'
   AND measure LIKE '%vaccine%'
   ORDER BY measure;`

**Regional Comparison Query:** "Compare regions vaccine data"
→ `SELECT region, measure, percent FROM vaccine_hesitancy_region 
   WHERE (measure LIKE '%unlikely%' OR measure LIKE '%declined%' OR measure LIKE '%hesitancy%')
   ORDER BY region, percent DESC;`

**Age Group Analysis Query:** "Which age group is most hesitant?"
→ `SELECT age_group, measure, percent FROM vaccine_hesitancy_age_group 
   WHERE (measure LIKE '%unlikely%' OR measure LIKE '%declined%' OR measure LIKE '%hesitancy%')
   ORDER BY percent DESC;`

**Broad Exploration Query:** "What vaccine data exists for employment status?"
→ `SELECT employment_status, measure, percent FROM vaccine_hesitancy_employment 
   WHERE measure LIKE '%vaccine%'
   ORDER BY employment_status, measure;`

**PROGRESSIVE FILTERING LOGIC:**
- **Primary**: Specific hesitancy measures + percent IS NOT NULL
- **Secondary**: Broader vaccine measures + percent IS NOT NULL  
- **Fallback**: All measures (including N/A) to show data availability
- **Never exclude**: Valid data just because some values are N/A
- **Always include**: Available data even if incomplete

Generate ONLY the SQL query, no explanation. Focus on hesitancy analysis and use the actual values from the database when possible!"""

    async def discover_available_data(self, table_name: str, dimension_column: str, dimension_value: str, ctx: Context) -> Dict[str, Any]:
        """Discover what data is actually available in a specific table for a dimension value"""
        try:
            # Escape table name and column names that might be SQL keywords
            escaped_table_name = f'"{table_name}"' if 'group' in table_name else table_name
            escaped_dimension_column = f'"{dimension_column}"' if dimension_column in ['group'] else dimension_column
            
            discovery_query = f"""
            SELECT measure, 
                   COUNT(*) as total_records,
                   COUNT(CASE WHEN percent IS NOT NULL AND percent != '' THEN 1 END) as records_with_data,
                   COUNT(CASE WHEN percent IS NULL OR percent = '' THEN 1 END) as records_with_na,
                   AVG(CASE WHEN percent IS NOT NULL AND percent != '' THEN percent END) as avg_percent
            FROM {escaped_table_name}
            WHERE {escaped_dimension_column} = '{dimension_value}'
            GROUP BY measure
            ORDER BY records_with_data DESC;
            """
            
            ctx.logger.info(f"🔍 [DISCOVERY] Checking available data for {dimension_value} in {table_name}")
            
            # Execute discovery query
            result = await self.execute_sql_with_retry(discovery_query, ctx, max_retries=2)
            
            if result and isinstance(result, list):
                ctx.logger.info(f"📊 [DISCOVERY] Found {len(result)} different measures for {dimension_value}")
                
                # Log detailed discovery info
                measures_with_data = [item for item in result if item.get('records_with_data', 0) > 0]
                measures_with_na_only = [item for item in result if item.get('records_with_data', 0) == 0]
                
                ctx.logger.info(f"✅ [DISCOVERY] {len(measures_with_data)} measures have actual data")
                ctx.logger.info(f"⚠️ [DISCOVERY] {len(measures_with_na_only)} measures have only N/A values")
                
                return {
                    "success": True,
                    "total_measures": len(result),
                    "measures_with_data": measures_with_data,
                    "measures_with_na_only": measures_with_na_only,
                    "has_data": len(measures_with_data) > 0
                }
            else:
                ctx.logger.warning(f"⚠️ [DISCOVERY] No data discovery results for {dimension_value}")
                return {
                    "success": False,
                    "total_measures": 0,
                    "measures_with_data": [],
                    "measures_with_na_only": [],
                    "has_data": False
                }
                
        except Exception as e:
            ctx.logger.error(f"❌ [DISCOVERY] Failed to discover data for {dimension_value}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "total_measures": 0,
                "measures_with_data": [],
                "measures_with_na_only": [],
                "has_data": False
            }

    async def execute_sql_with_retry(self, sql_query: str, ctx: Context, max_retries: int = 3) -> Any:
        """Execute SQL query with retry logic"""
        for attempt in range(max_retries):
            try:
                result = await self.mcp_client.call_tool("execute_sql", {
                    "project_id": SUPABASE_PROJECT_ID,
                    "query": sql_query
                }, ctx)
                
                # Process the result
                content = result.content[0].text if isinstance(result.content, list) else result.content
                return json.loads(content)
                
            except Exception as e:
                if attempt == max_retries - 1:
                    ctx.logger.error(f"❌ SQL execution failed after {max_retries} attempts: {str(e)}")
                    raise e
                else:
                    ctx.logger.warning(f"⚠️ SQL attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    await asyncio.sleep(1)  # Brief delay before retry
        
        return None

    async def query_ons_data(self, refined_query: str, ctx: Context) -> Dict[str, Any]:
        """ONS Tool: Specialized for vaccine hesitancy demographic data with PROGRESSIVE FILTERING"""
        
        try:
            # Step 1: Discover actual database schema
            ctx.logger.info("🔍 Discovering database schema for dynamic prompt...")
            schema_info = await self.discover_database_schema(ctx)
            ctx.logger.info(f"✅ Schema discovery complete. Found {len(schema_info.get('valid_measures', []))} measures, {len(schema_info.get('valid_age_groups', []))} age groups")
            
            # Step 2: Generate dynamic prompt based on discovered schema
            dynamic_ons_prompt = self.generate_dynamic_ons_prompt(schema_info)
            
            # Step 3: Progressive query execution with data discovery
            result_data = await self.execute_ons_query_progressive(refined_query, dynamic_ons_prompt, ctx)
            
            if result_data.get("success"):
                return result_data
            else:
                # If progressive approach fails, return error
                return {"error": result_data.get("error", "Query execution failed"), "query": refined_query, "source": "ONS"}
            
        except Exception as e:
            ctx.logger.error(f"❌ [ONS] Complete failure for query: '{refined_query}'")
            ctx.logger.error(f"🔍 [ONS] Error details: {str(e)}")
            ctx.logger.error(f"📝 [ONS] Query context: {refined_query}")
            return {"error": str(e), "query": refined_query, "source": "ONS"}

    async def execute_ons_query_progressive(self, refined_query: str, dynamic_ons_prompt: str, ctx: Context) -> Dict[str, Any]:
        """Execute ONS query with progressive filtering - start restrictive, then broaden if needed"""
        
        # Step 1: Try primary query (restrictive)
        ctx.logger.info(f"🔍 [PROGRESSIVE] Attempting primary query for: '{refined_query}'")
        primary_result = await self.execute_single_ons_query(refined_query, dynamic_ons_prompt, ctx, query_type="primary")
        
        if primary_result.get("success") and primary_result.get("row_count", 0) >= 5:
            ctx.logger.info(f"✅ [PROGRESSIVE] Primary query successful: {primary_result.get('row_count')} rows")
            return primary_result
        
        # Step 2: Check if data exists using discovery
        ctx.logger.info(f"🔍 [PROGRESSIVE] Primary query returned {primary_result.get('row_count', 0)} rows, checking data availability...")
        
        # Try to detect table and dimension from the query
        table_info = self.detect_table_and_dimension(refined_query, primary_result.get("sql_query", ""))
        
        if table_info:
            discovery_result = await self.discover_available_data(
                table_info["table"], 
                table_info["dimension_column"], 
                table_info["dimension_value"], 
                ctx
            )
            
            if discovery_result.get("has_data"):
                ctx.logger.info(f"✅ [PROGRESSIVE] Data exists for {table_info['dimension_value']}, trying fallback query...")
                
                # Step 3: Try fallback query (broader)
                fallback_result = await self.execute_single_ons_query(refined_query, dynamic_ons_prompt, ctx, query_type="fallback")
                
                if fallback_result.get("success") and fallback_result.get("row_count", 0) > 0:
                    ctx.logger.info(f"✅ [PROGRESSIVE] Fallback query successful: {fallback_result.get('row_count')} rows")
                    # Add discovery info to the result
                    fallback_result["discovery_info"] = discovery_result
                    return fallback_result
        
        # Step 4: If still no data, try simplified query
        ctx.logger.info(f"🔍 [PROGRESSIVE] Trying simplified query as final fallback...")
        simplified_result = await self.execute_single_ons_query(refined_query, dynamic_ons_prompt, ctx, query_type="simplified")
        
        if simplified_result.get("success"):
            ctx.logger.info(f"✅ [PROGRESSIVE] Simplified query successful: {simplified_result.get('row_count')} rows")
            return simplified_result
        
        # Step 5: All approaches failed
        ctx.logger.error(f"❌ [PROGRESSIVE] All query approaches failed for: '{refined_query}'")
        return {
            "success": False,
            "error": "No data found after trying multiple query approaches",
            "query": refined_query,
            "attempts": {
                "primary": primary_result,
                "simplified": simplified_result
            }
        }

    def detect_table_and_dimension(self, refined_query: str, sql_query: str) -> Dict[str, str]:
        """Detect table and dimension information from query for data discovery"""
        try:
            # Simple detection logic based on keywords in the query
            query_lower = refined_query.lower()
            sql_lower = sql_query.lower() if sql_query else ""
            
            # Table detection
            table_mapping = {
                "buddhist": ("vaccine_hesitancy_religion", "religion", "Buddhist"),
                "religion": ("vaccine_hesitancy_religion", "religion", None),
                "age": ("vaccine_hesitancy_age_group", "age_group", None),
                "region": ("vaccine_hesitancy_region", "region", None),
                "disability": ("vaccine_hesitancy_disability", "disability_status", None),
                "employment": ("vaccine_hesitancy_employment", "employment_status", None),
                "ethnicity": ("vaccine_hesitancy_ethnicity", "ethnicity", None),
                "sex": ("vaccine_hesitancy_sex", "sex", None),
                "gender": ("vaccine_hesitancy_sex", "sex", None)
            }
            
            # Check for specific keywords
            for keyword, (table, column, value) in table_mapping.items():
                if keyword in query_lower or keyword in sql_lower:
                    # Try to extract specific value from SQL if not predefined
                    if not value and "=" in sql_lower:
                        # Simple extraction: look for column = 'value' pattern
                        import re
                        pattern = rf"{column}\s*=\s*'([^']+)'"
                        match = re.search(pattern, sql_lower)
                        if match:
                            value = match.group(1)
                    
                    if value:
                        return {
                            "table": table,
                            "dimension_column": column,
                            "dimension_value": value
                        }
            
            return None
            
        except Exception as e:
            return None

    async def execute_single_ons_query(self, refined_query: str, dynamic_ons_prompt: str, ctx: Context, query_type: str = "primary") -> Dict[str, Any]:
        """Execute a single ONS query with specific approach"""
        
        try:
            # Modify prompt based on query type
            if query_type == "fallback":
                modified_prompt = dynamic_ons_prompt.replace(
                    "WHERE percent IS NOT NULL", 
                    "-- Include all data, even N/A values for completeness"
                ).replace(
                    "Generate ONLY the SQL query", 
                    "Generate broader SQL query including all vaccine measures. Generate ONLY the SQL query"
                )
            elif query_type == "simplified":
                modified_prompt = """Generate a simple SQL query for vaccine hesitancy data. 
                
                Available tables:
                - vaccine_hesitancy_religion (religion, measure, percent)
                - vaccine_hesitancy_age_group (age_group, measure, percent) 
                - vaccine_hesitancy_region (region, measure, percent)
                - vaccine_hesitancy_disability (disability_status, measure, percent)
                
                Use basic SELECT, FROM, WHERE structure only. 
                Include measures with ANY vaccine-related content.
                Focus on single table queries - never JOIN tables.
                Generate ONLY the SQL query, no explanation."""
            else:
                modified_prompt = dynamic_ons_prompt
            
            # Generate SQL using ASI1
            payload = {
                "model": "asi1-mini",
                "messages": [
                    {"role": "system", "content": modified_prompt},
                    {"role": "user", "content": f"Generate SQL query for: {refined_query}"}
                ],
                "temperature": 0.1,
                "max_tokens": 400 if query_type != "simplified" else 200
            }
            
            response = requests.post(ASI1_URL, headers=ASI1_HEADERS, json=payload)
            if response.status_code != 200:
                ctx.logger.error(f"❌ ASI1 API error: {response.status_code} for {query_type} query: {refined_query}")
                return {"success": False, "error": f"ASI1 API error: {response.status_code}"}
            
            sql_query = response.json()["choices"][0]["message"]["content"].strip()
            
            # Clean up the SQL query
            if "```sql" in sql_query:
                sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql_query:
                sql_query = sql_query.split("```")[1].strip()
            
            sql_query = sql_query.replace("`", "").replace(f"{SUPABASE_PROJECT_ID}.", "")
            
            # Log the SQL query before execution
            ctx.logger.info(f"🔍 [SQL] Executing {query_type} ONS query for: '{refined_query}'")
            ctx.logger.info(f"📝 [SQL] Generated SQL: {sql_query}")
            
            # Execute SQL with timing
            start_time = datetime.now(timezone.utc)
            
            result = await self.execute_sql_with_retry(sql_query, ctx, max_retries=2)
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            ctx.logger.info(f"⏱️ [SQL] Execution completed in {execution_time:.3f}s")
            
            # Process the result
            data = result if result else []
            
            # Log successful execution details
            row_count = len(data) if isinstance(data, list) else 0
            ctx.logger.info(f"✅ [SQL] {query_type.title()} query success: {row_count} rows returned")
            
            # Log data sample for debugging (first 3 rows)
            if row_count > 0:
                sample_data = data[:3] if isinstance(data, list) else [data]
                ctx.logger.info(f"📊 [SQL] Data sample (first {len(sample_data)} rows): {sample_data}")
                
                # Log basic data statistics
                if isinstance(data, list) and len(data) > 0:
                    first_row = data[0]
                    columns = list(first_row.keys()) if isinstance(first_row, dict) else []
                    ctx.logger.info(f"📋 [SQL] Columns returned: {columns}")
            
            return {
                "success": True,
                "data": data,
                "sql_query": sql_query,
                "source": "ONS",
                "execution_time": execution_time,
                "row_count": row_count,
                "query_type": query_type
            }
            
        except Exception as e:
            ctx.logger.error(f"❌ [SQL] {query_type.title()} query failed for: '{refined_query}' - {str(e)}")
            return {"success": False, "error": str(e), "query_type": query_type}

    async def query_twitter_data(self, refined_query: str, ctx: Context) -> Dict[str, Any]:
        """Twitter Tool: Intelligent SQL generation for vaccine_tweets analysis with ASI1"""
        
        twitter_system_prompt = """You are a Twitter vaccine sentiment data specialist. Generate intelligent SQL queries for this table:

AVAILABLE TWITTER TABLE & COLUMNS:
vaccine_tweets: tweet_id, content, author_username, author_name, author_profile_image, created_at, fetched_at, likes, retweets, replies, sentiment, impact_score, engagement_score

ANALYSIS CAPABILITIES:
- Sentiment Analysis: sentiment values are 'positive', 'negative', 'neutral'
- Timeline Analysis: Use DATE_TRUNC with created_at for trends over time  
- Engagement Analysis: likes, retweets, replies, impact_score, engagement_score
- User Analysis: GROUP BY author_username, author_name for user insights
- Content Analysis: Filter by content patterns, use LEFT(content, N) for previews
- Time Filtering: WHERE created_at >= NOW() - INTERVAL 'X days/hours/weeks'

QUERY STRATEGY:
1. Timeline queries (timeline, trends, over time): Use DATE_TRUNC, GROUP BY sentiment, show sentiment distribution over time
2. Top content queries (viral, popular, top tweets): ORDER BY engagement metrics DESC
3. User/influencer queries (authors, users, influencers): GROUP BY author, SUM/COUNT engagement  
4. Sentiment queries (distribution, breakdown): GROUP BY sentiment with percentages across entire dataset
5. General queries: Smart SELECT with appropriate WHERE clauses

SQL RULES:
- Use simple table names: vaccine_tweets (NO backticks, NO project prefixes)
- Use ROUND(AVG(impact_score)::numeric, 2) for PostgreSQL compatibility
- Always include reasonable LIMIT to avoid large results
- Use appropriate time filters for recent data
- For timeline: DATE_TRUNC('day', created_at) or 'hour', 'week'

CRITICAL - PERCENTAGE CALCULATIONS:
- For overall sentiment distribution: ROUND((COUNT(*) * 100.0 / SUM(COUNT(*)) OVER())::numeric, 2)
- For daily sentiment within each day: ROUND((COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(PARTITION BY DATE_TRUNC('day', created_at)))::numeric, 2)
- For timeline queries, usually want OVERALL percentages, NOT daily percentages

CORRECT EXAMPLES:

Timeline with overall sentiment percentages:
SELECT DATE_TRUNC('day', created_at) as date, sentiment, COUNT(*) as tweet_count, ROUND((COUNT(*) * 100.0 / SUM(COUNT(*)) OVER())::numeric, 2) as percentage FROM vaccine_tweets WHERE created_at >= NOW() - INTERVAL '30 days' GROUP BY date, sentiment ORDER BY date, sentiment;

Sentiment distribution:
SELECT sentiment, COUNT(*) as count, ROUND((COUNT(*) * 100.0 / SUM(COUNT(*)) OVER())::numeric, 2) as percentage FROM vaccine_tweets GROUP BY sentiment;

Top viral tweets:
SELECT content, author_username, likes, retweets, replies FROM vaccine_tweets ORDER BY (likes + retweets + replies) DESC LIMIT 15;

Generate ONLY the SQL query, no explanation. Be intelligent about what analysis type is needed!"""

        try:
            payload = {
                "model": "asi1-mini", 
                "messages": [
                    {"role": "system", "content": twitter_system_prompt},
                    {"role": "user", "content": f"Generate SQL query for: {refined_query}"}
                ],
                "temperature": 0.1,
                "max_tokens": 200
            }
            
            response = requests.post(ASI1_URL, headers=ASI1_HEADERS, json=payload)
            if response.status_code != 200:
                ctx.logger.error(f"❌ [Twitter] ASI1 API error: {response.status_code} for query: {refined_query}")
                return {"error": f"ASI1 API error: {response.status_code}", "query": refined_query}
            
            sql_query = response.json()["choices"][0]["message"]["content"].strip()
            
            # Clean up the SQL query - remove any markdown formatting or extra text
            if "```sql" in sql_query:
                sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql_query:
                sql_query = sql_query.split("```")[1].strip()
            
            # Remove any backticks or project prefixes that might slip through
            sql_query = sql_query.replace("`", "").replace(f"{SUPABASE_PROJECT_ID}.", "")
            
            # Log the SQL query before execution
            ctx.logger.info(f"🔍 [Twitter SQL] Executing Twitter query for: '{refined_query}'")
            ctx.logger.info(f"📝 [Twitter SQL] Generated SQL: {sql_query}")
            
            # Execute SQL with timing
            start_time = datetime.now(timezone.utc)
            try:
                result = await self.mcp_client.call_tool("execute_sql", {
                    "project_id": SUPABASE_PROJECT_ID,
                    "query": sql_query
                }, ctx)
                
                execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                ctx.logger.info(f"⏱️ [Twitter SQL] Execution completed in {execution_time:.3f}s")
                
                # Process the result
                content = result.content[0].text if isinstance(result.content, list) else result.content
                data = json.loads(content)
                
                # Log successful execution details
                row_count = len(data) if isinstance(data, list) else 0
                ctx.logger.info(f"✅ [Twitter SQL] Success: {row_count} rows returned")
                
                # Log data sample for debugging (first 3 rows)
                if row_count > 0:
                    sample_data = data[:3] if isinstance(data, list) else [data]
                    ctx.logger.info(f"📊 [Twitter SQL] Data sample (first {len(sample_data)} rows): {sample_data}")
                    
                    # Log basic data statistics
                    if isinstance(data, list) and len(data) > 0:
                        first_row = data[0]
                        columns = list(first_row.keys()) if isinstance(first_row, dict) else []
                        ctx.logger.info(f"📋 [Twitter SQL] Columns returned: {columns}")
                        
                        # Log Twitter-specific insights
                        if 'sentiment' in columns:
                            sentiment_counts = {}
                            for row in data:
                                sentiment = row.get('sentiment', 'unknown')
                                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
                            ctx.logger.info(f"😊 [Twitter SQL] Sentiment distribution: {sentiment_counts}")
                        
                        if 'engagement_score' in columns:
                            engagement_scores = [row.get('engagement_score', 0) for row in data if row.get('engagement_score')]
                            if engagement_scores:
                                avg_engagement = sum(engagement_scores) / len(engagement_scores)
                                ctx.logger.info(f"📈 [Twitter SQL] Average engagement score: {avg_engagement:.2f}")
                        
                        # Log data quality info
                        null_counts = {}
                        if isinstance(data, list):
                            for col in columns:
                                null_count = sum(1 for row in data if row.get(col) is None or row.get(col) == '')
                                if null_count > 0:
                                    null_counts[col] = null_count
                        
                        if null_counts:
                            ctx.logger.warning(f"⚠️ [Twitter SQL] Data quality: {null_counts} null/empty values found")
                
                return {
                    'success': True,
                    'data': data,
                    'sql_query': sql_query,
                    'source': 'Twitter',
                    'execution_time': execution_time,
                    'row_count': row_count
                }
                
            except Exception as sql_error:
                execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                ctx.logger.error(f"❌ [Twitter SQL] Execution failed after {execution_time:.3f}s: {str(sql_error)}")
                ctx.logger.error(f"🔍 [Twitter SQL] Failed query: {sql_query}")
                ctx.logger.error(f"📝 [Twitter SQL] User query: {refined_query}")
                raise sql_error
            
        except Exception as e:
            ctx.logger.error(f"❌ [Twitter] Complete failure for query: '{refined_query}'")
            ctx.logger.error(f"🔍 [Twitter] Error details: {str(e)}")
            ctx.logger.error(f"📝 [Twitter] Query context: {refined_query}")
            return {"error": str(e), "query": refined_query, "source": "Twitter"}

    async def check_and_route_query(self, user_query: str, ctx: Context) -> Dict[str, Any]:
        """Built-in checker logic to determine routing and refine queries"""
        
        # Simple fallback routing for common patterns before ASI1 call
        query_lower = user_query.lower()
        
        # ONS-specific keywords
        ons_keywords = [
            'gender', 'sex', 'age', 'ethnicity', 'religion', 'disability', 'employment', 
            'demographic', 'region', 'health condition', 'barriers', 'reasons', 'trends',
            'imd', 'quintile', 'caregiver', 'household', 'expense', 'affordability'
        ]
        
        # Twitter-specific keywords (enhanced for intelligent routing)
        twitter_keywords = [
            'sentiment', 'social media', 'twitter', 'engagement', 'tweets', 'posts',
            'opinion', 'discussion', 'conversation', 'viral', 'trending', 'timeline',
            'influencer', 'top tweets', 'popular', 'retweets', 'likes', 'authors',
            'users', 'distribution', 'breakdown', 'over time', 'recent', 'daily',
            'weekly', 'hourly', 'social', 'online', 'digital', 'content', 'influence'
        ]
        
        # Check for direct ONS matches
        if any(keyword in query_lower for keyword in ons_keywords):
            return {
                "routing": "ons_only",
                "refined_query": user_query,
                "explanation": "Query contains demographic/statistical terms available in ONS data"
            }
        
        # Check for direct Twitter matches
        if any(keyword in query_lower for keyword in twitter_keywords):
            return {
                "routing": "twitter_only", 
                "refined_query": user_query,
                "explanation": "Query contains social media/sentiment terms available in Twitter data"
            }
        
        # Try ASI1 for more complex routing
        checker_prompt = f"""Analyze this vaccine hesitancy query and determine routing:

Query: "{user_query}"

**AVAILABLE DATA:**
- ONS: Demographics (age, sex, ethnicity, religion, disability, employment, etc.), trends, barriers, reasons for hesitancy
- Twitter: Sentiment analysis, timeline trends, engagement metrics, user/influencer analysis, content analysis, viral tweets

**ROUTING OPTIONS:**
1. "ons_only" - Query needs only ONS demographic/statistical data
2. "twitter_only" - Query needs only Twitter sentiment/social data  
3. "both" - Query needs comparative/combined analysis
4. "unavailable" - Cannot be answered with available data

**OUTPUT FORMAT:**
{{
  "routing": "ons_only|twitter_only|both|unavailable",
  "refined_query": "optimized query for tools",
  "explanation": "brief reason for routing decision"
}}

Respond with valid JSON only."""

        try:
            payload = {
                "model": "asi1-mini",
                "messages": [
                    {"role": "system", "content": checker_prompt},
                    {"role": "user", "content": user_query}
                ],
                "temperature": 0.1,
                "max_tokens": 150
            }
            
            ctx.logger.info(f"Sending checker query to ASI1: {user_query}")
            response = requests.post(ASI1_URL, headers=ASI1_HEADERS, json=payload)
            
            if response.status_code != 200:
                ctx.logger.error(f"ASI1 API error: {response.status_code} - {response.text}")
                # Fallback to ONS for general vaccine hesitancy queries
                return {
                    "routing": "ons_only",
                    "refined_query": user_query,
                    "explanation": "Fallback routing due to ASI1 API error"
                }
            
            result_text = response.json()["choices"][0]["message"]["content"].strip()
            ctx.logger.info(f"ASI1 checker response: {result_text}")
            
            # Try to parse JSON response
            try:
                parsed_result = json.loads(result_text)
                return parsed_result
            except json.JSONDecodeError as json_err:
                ctx.logger.error(f"JSON parsing error: {json_err}")
                ctx.logger.error(f"Raw ASI1 response: {result_text}")
                
                # Fallback parsing for non-JSON responses
                if "ons" in result_text.lower() and "twitter" not in result_text.lower():
                    return {
                        "routing": "ons_only",
                        "refined_query": user_query,
                        "explanation": "Parsed from non-JSON ASI1 response indicating ONS data needed"
                    }
                elif "twitter" in result_text.lower() and "ons" not in result_text.lower():
                    return {
                        "routing": "twitter_only",
                        "refined_query": user_query,
                        "explanation": "Parsed from non-JSON ASI1 response indicating Twitter data needed"
                    }
                elif "both" in result_text.lower():
                    return {
                        "routing": "both",
                        "refined_query": user_query,
                        "explanation": "Parsed from non-JSON ASI1 response indicating both data sources needed"
                    }
                else:
                    # Final fallback - try ONS for vaccine hesitancy queries
                    return {
                        "routing": "ons_only",
                        "refined_query": user_query,
                        "explanation": "Final fallback routing to ONS for vaccine hesitancy query"
                    }
            
        except Exception as e:
            ctx.logger.error(f"Checker error: {str(e)}")
            # Robust fallback routing
            if "vaccine" in query_lower or "hesitancy" in query_lower:
                return {
                    "routing": "ons_only",
                    "refined_query": user_query,
                    "explanation": "Exception fallback routing to ONS for vaccine hesitancy query"
                }
            else:
                return {
                    "routing": "unavailable", 
                    "error": "Unable to analyze query and no suitable fallback found"
                }

    # ============================================================================
    # ENHANCED DATA ANALYSIS METHODS
    # ============================================================================
    
    def categorize_measure_pattern(self, measure_name: str) -> tuple[str, float]:
        """
        Categorize vaccine measures using pattern matching with confidence scoring
        Returns: (category, confidence_score)
        """
        
        if not measure_name:
            return 'unknown', 0.0
            
        measure_lower = measure_name.lower().strip()
        
        # HIGH CONFIDENCE PATTERNS (exact matches or very specific patterns)
        high_confidence_patterns = {
            'acceptance': [
                'have received a vaccine',
                'positive vaccine sentiment',
                'would be very or fairly likely to have a vaccine if offered',
                'received vaccine',
                'vaccinated',
                'positive sentiment'
            ],
            'hesitancy': [
                'have been offered a vaccine but declined the offer',
                'would be very or fairly unlikely to have a vaccine if offered',
                'vaccine hesitancy',
                'declined the offer',
                'unlikely to have a vaccine'
            ],
            'neutral': [
                'are neither likely nor unlikely to have a vaccine if offered',
                'don\'t know',
                'prefer not to say',
                'neither likely nor unlikely'
            ],
            'waiting': [
                'have been offered a vaccine and waiting to be vaccinated',
                'waiting to be vaccinated'
            ]
        }
        
        # Check exact matches first (highest confidence)
        for category, patterns in high_confidence_patterns.items():
            for pattern in patterns:
                if pattern in measure_lower:
                    return category, 0.95  # 95% confidence for exact matches
        
        # MEDIUM CONFIDENCE PATTERNS (keyword-based)
        medium_confidence_patterns = {
            'acceptance': [
                'received', 'likely', 'positive', 'willing', 'uptake', 'completed', 'doses'
            ],
            'hesitancy': [
                'declined', 'refused', 'unlikely', 'negative', 'hesitant', 'reluctant', 'concern'
            ],
            'neutral': [
                'neither', 'uncertain', 'unsure', 'maybe', 'possibly'
            ],
            'waiting': [
                'waiting', 'offered', 'pending', 'scheduled', 'appointment'
            ]
        }
        
        # Check keyword matches (medium confidence)
        for category, keywords in medium_confidence_patterns.items():
            if any(keyword in measure_lower for keyword in keywords):
                return category, 0.75  # 75% confidence for keyword matches
        
        return 'unknown', 0.0  # No pattern matched

    def _calculate_category_rate(self, category_data: List[Dict]) -> Dict[str, Any]:
        """Calculate average rate for a category with quality metrics"""
        valid_measures = [row for row in category_data if row.get('percent') and str(row.get('percent')).lower() not in ['nan', 'n/a', '']]
        
        if not valid_measures:
            return {
                'rate': None,
                'count': 0,
                'data_available': False,
                'message': 'No data available for this category',
                'measures': []
            }
        
        percentages = [float(row['percent']) for row in valid_measures]
        avg_rate = sum(percentages) / len(percentages)
        
        return {
            'rate': round(avg_rate, 2),
            'count': len(valid_measures),
            'data_available': True,
            'message': f'Based on {len(valid_measures)} measures with data',
            'measures': [row.get('measure', 'Unknown') for row in valid_measures]
        }

    def _calculate_overall_confidence(self, categorized_measures: Dict) -> float:
        """Calculate overall confidence in categorization"""
        total_measures = sum(len(measures) for measures in categorized_measures.values())
        if total_measures == 0:
            return 0.0
        
        # For this implementation, return a reasonable confidence based on categorization success
        unknown_count = len(categorized_measures.get('unknown', []))
        categorized_count = total_measures - unknown_count
        
        if total_measures == 0:
            return 0.0
        
        return round((categorized_count / total_measures) * 0.9, 2)  # Max 90% confidence

    async def analyze_hesitancy_reasons(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze hesitancy reasons data for patterns and insights"""
        
        # Filter out null values and technical artifacts
        valid_data = [row for row in data if row.get('percent') and row.get('measure') and 
                     str(row.get('measure', '')).lower() != 'nan' and 
                     row.get('measure') != row.get('block')]
        
        if not valid_data:
            return {'type': 'no_valid_data', 'data': data}
        
        # Sort by percentage to find top factors
        sorted_data = sorted(valid_data, key=lambda x: float(x.get('percent', 0)), reverse=True)
        
        # Extract key insights
        top_factors = sorted_data[:5]  # Top 5 reasons
        total_weighted = sum(float(row.get('weighted_count', 0)) for row in valid_data if row.get('weighted_count'))
        
        return {
            'type': 'hesitancy_reasons',
            'top_factors': top_factors,
            'total_weighted': total_weighted,
            'factor_count': len(valid_data),
            'data_period': next((row.get('period') for row in data if row.get('period')), 'Unknown'),
            'all_factors': sorted_data
        }
    
    async def analyze_demographic_patterns(self, data: List[Dict], dimension: str = None) -> Dict[str, Any]:
        """Analyze demographic data with intelligent measure categorization"""
        
        if not data:
            return {'type': 'no_data', 'error': 'Empty dataset provided'}
        
        # Get all columns from the first row
        available_columns = list(data[0].keys()) if data else []
        
        # Auto-detect dimension if not provided
        if not dimension:
            demographic_indicators = ['age', 'sex', 'gender', 'ethnicity', 'religion', 'region', 'disability', 
                                    'employment', 'quintile', 'health', 'group', 'band', 'status', 'condition']
            
            potential_dimensions = []
            for col in available_columns:
                col_lower = str(col).lower()
                if any(indicator in col_lower for indicator in demographic_indicators):
                    if col_lower not in ['percent', 'percentage', 'measure', 'count', 'size', 'date', 'time', 'wave']:
                        potential_dimensions.append(col)
            
            if potential_dimensions:
                dimension = potential_dimensions[0]
            else:
                numeric_indicators = ['percent', 'count', 'size', 'score', 'rate', 'value']
                categorical_columns = [col for col in available_columns 
                                     if not any(indicator in str(col).lower() for indicator in numeric_indicators)]
                dimension = categorical_columns[0] if categorical_columns else available_columns[0]
        
        # Group by the detected dimension
        grouped_data = {}
        total_rows = len(data)
        
        for row in data:
            key = row.get(dimension, 'Unknown')
            if key and str(key).lower() not in ['nan', 'null', '']:
                if key not in grouped_data:
                    grouped_data[key] = []
                grouped_data[key].append(row)
        
        if not grouped_data:
            return {
                'type': 'no_grouped_data', 
                'error': f'No valid groups found for dimension: {dimension}',
                'available_columns': available_columns,
                'dimension_used': dimension,
                'total_rows': total_rows
            }
        
        # Analyze each group with intelligent measure categorization
        analysis_results = {}
        for key, rows in grouped_data.items():
            categorized_measures = {
                'acceptance': [],
                'hesitancy': [],
                'neutral': [],
                'waiting': [],
                'unknown': []
            }
            
            # Categorize each measure intelligently
            for row in rows:
                measure_name = row.get('measure', '')
                if measure_name:
                    category, confidence = self.categorize_measure_pattern(measure_name)
                    categorized_measures[category].append(row)
            
            # Calculate rates for each category
            analysis_results[key] = {
                'acceptance_rate': self._calculate_category_rate(categorized_measures['acceptance']),
                'hesitancy_rate': self._calculate_category_rate(categorized_measures['hesitancy']),
                'neutral_rate': self._calculate_category_rate(categorized_measures['neutral']),
                'waiting_rate': self._calculate_category_rate(categorized_measures['waiting']),
                'data_quality': {
                    'total_measures': len(rows),
                    'measures_with_data': sum(1 for row in rows if row.get('percent') and str(row.get('percent')).lower() not in ['nan', 'n/a', '']),
                    'categorization_confidence': self._calculate_overall_confidence(categorized_measures)
                },
                'detailed_breakdown': {
                    cat: [row.get('measure', 'Unknown') for row in measures] 
                    for cat, measures in categorized_measures.items()
                }
            }
        
        return {
            'type': 'demographic_patterns_enhanced',
            'dimension': dimension,
            'dimension_display': dimension.replace('_', ' ').title(),
            'analysis_results': analysis_results,
            'available_columns': available_columns,
            'total_data_points': total_rows
        }
    
    async def analyze_twitter_patterns(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze Twitter data for sentiment and engagement patterns"""
        
        if not data:
            return {'type': 'no_twitter_data'}
        
        # Sentiment analysis
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        engagement_scores = []
        
        for row in data:
            sentiment = str(row.get('sentiment', '')).lower()
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
            
            if row.get('engagement_score'):
                engagement_scores.append(float(row.get('engagement_score', 0)))
        
        total_tweets = sum(sentiment_counts.values())
        sentiment_percentages = {k: (v/total_tweets)*100 if total_tweets > 0 else 0 
                               for k, v in sentiment_counts.items()}
        
        avg_engagement = sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0
        
        return {
            'type': 'twitter_patterns',
            'sentiment_distribution': sentiment_percentages,
            'total_tweets': total_tweets,
            'avg_engagement': avg_engagement,
            'top_sentiment': max(sentiment_percentages, key=sentiment_percentages.get)
        }
    
    async def generate_insights_with_asi1(self, data: List[Dict], analysis: Dict, user_query: str, source: str) -> str:
        """Use ASI1 to generate conversational insights from analyzed data"""
        
        # Create context-aware prompt based on analysis type
        if analysis.get('type') == 'hesitancy_reasons':
            context = f"""Top vaccine hesitancy factors:
{', '.join([f"{factor.get('measure', 'Unknown')}: {factor.get('percent', 0)}%" for factor in analysis.get('top_factors', [])[:3]])}
Data period: {analysis.get('data_period', 'Unknown')}
Total factors analyzed: {analysis.get('factor_count', 0)}"""
        
        elif analysis.get('type') == 'demographic_patterns_enhanced':
            analysis_results = analysis.get('analysis_results', {})
            dimension_display = analysis.get('dimension_display', analysis.get('dimension', 'unknown'))
            
            # Get the first (and likely only) group's results
            if analysis_results:
                group_name = list(analysis_results.keys())[0]
                group_data = analysis_results[group_name]
                
                acceptance_rate = group_data.get('acceptance_rate', {})
                hesitancy_rate = group_data.get('hesitancy_rate', {})
                data_quality = group_data.get('data_quality', {})
                
                context_parts = [f"Demographic analysis by {dimension_display}:"]
                
                # Add acceptance data if available
                if acceptance_rate.get('data_available'):
                    context_parts.append(f"Vaccine acceptance rate: {acceptance_rate['rate']}% ({acceptance_rate['message']})")
                    if acceptance_rate.get('measures'):
                        context_parts.append(f"Acceptance measures: {', '.join(acceptance_rate['measures'][:2])}")
                
                # Add hesitancy data if available  
                if hesitancy_rate.get('data_available'):
                    context_parts.append(f"Vaccine hesitancy rate: {hesitancy_rate['rate']}% ({hesitancy_rate['message']})")
                    if hesitancy_rate.get('measures'):
                        context_parts.append(f"Hesitancy measures: {', '.join(hesitancy_rate['measures'][:2])}")
                else:
                    context_parts.append("Vaccine hesitancy rate: Data not available (N/A)")
                
                # Add data quality info
                context_parts.append(f"Total measures: {data_quality.get('total_measures', 0)}")
                context_parts.append(f"Measures with data: {data_quality.get('measures_with_data', 0)}")
                context_parts.append(f"Categorization confidence: {data_quality.get('categorization_confidence', 0)*100:.0f}%")
                
                context = "\n".join(context_parts)
            else:
                context = f"Demographic analysis by {dimension_display}: No data available for analysis"
        
        elif analysis.get('type') == 'twitter_patterns':
            context = f"""Social media sentiment analysis:
Positive: {analysis.get('sentiment_distribution', {}).get('positive', 0):.1f}%
Negative: {analysis.get('sentiment_distribution', {}).get('negative', 0):.1f}%
Neutral: {analysis.get('sentiment_distribution', {}).get('neutral', 0):.1f}%
Total tweets: {analysis.get('total_tweets', 0)}
Dominant sentiment: {analysis.get('top_sentiment', 'unknown')}"""
        
        else:
            context = f"General data analysis with {len(data)} records from {source}"
        
        insight_prompt = f"""You are a public health expert analyzing vaccine hesitancy data. 

User Question: "{user_query}"
Data Source: {source}
Data Summary: {context}

Your task: Create a conversational, expert-level response that:
1. Directly answers the user's question with specific findings
2. Explains what the numbers mean in practical terms
3. Identifies key patterns and their significance
4. Provides public health context and implications
5. Uses professional but accessible language

Format your response like this:
- Start with "Based on [data source/period], here are the key findings:"
- Use bullet points for main findings with specific percentages
- Include a "What This Means:" section with interpretation
- Keep technical jargon to a minimum
- Focus on actionable insights

Write as a knowledgeable public health analyst, not a database report."""
        
        try:
            payload = {
                "model": "asi1-mini",
                "messages": [
                    {"role": "system", "content": insight_prompt},
                    {"role": "user", "content": f"Generate expert insights for: {user_query}"}
                ],
                "temperature": 0.3,
                "max_tokens": 500
            }
            
            response = requests.post(ASI1_URL, headers=ASI1_HEADERS, json=payload)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            else:
                return self._fallback_insight_generation(data, analysis, user_query, source)
                
        except Exception as e:
            return self._fallback_insight_generation(data, analysis, user_query, source)
    
    def _fallback_insight_generation(self, data: List[Dict], analysis: Dict, user_query: str, source: str) -> str:
        """Generate insights when ASI1 is unavailable"""
        
        if analysis.get('type') == 'hesitancy_reasons':
            top_factors = analysis.get('top_factors', [])
            if top_factors:
                insights = ["Based on the latest vaccine hesitancy data, here are the key findings:\n"]
                insights.append("**Top Reasons for Vaccine Hesitancy:**")
                for i, factor in enumerate(top_factors[:3], 1):
                    measure = factor.get('measure', 'Unknown reason')
                    percent = factor.get('percent', 0)
                    insights.append(f"• **{measure}**: {percent}% of hesitant individuals")
                
                insights.append("\n**What This Means:**")
                insights.append("The data shows that safety concerns are the primary barrier to vaccination, ")
                insights.append("suggesting that public health messaging should focus on addressing these specific worries.")
                
                return "\n".join(insights)
        
        elif analysis.get('type') == 'demographic_patterns_enhanced':
            analysis_results = analysis.get('analysis_results', {})
            dimension_display = analysis.get('dimension_display', analysis.get('dimension', 'demographic groups'))
            
            if analysis_results:
                group_name = list(analysis_results.keys())[0]
                group_data = analysis_results[group_name]
                
                acceptance_rate = group_data.get('acceptance_rate', {})
                hesitancy_rate = group_data.get('hesitancy_rate', {})
                neutral_rate = group_data.get('neutral_rate', {})
                waiting_rate = group_data.get('waiting_rate', {})
                data_quality = group_data.get('data_quality', {})
                
                insights = [f"Based on demographic analysis by {dimension_display}, here are the key findings:\n"]
                
                # Report acceptance data
                if acceptance_rate.get('data_available'):
                    insights.append("**Vaccine Acceptance Data:**")
                    insights.append(f"• **{acceptance_rate['rate']}% acceptance rate** ({acceptance_rate['message']})")
                    if acceptance_rate.get('measures'):
                        insights.append(f"• **Measures**: {', '.join(acceptance_rate['measures'][:2])}")
                
                # Report hesitancy data  
                if hesitancy_rate.get('data_available'):
                    insights.append("\n**Vaccine Hesitancy Data:**")
                    insights.append(f"• **{hesitancy_rate['rate']}% hesitancy rate** ({hesitancy_rate['message']})")
                    if hesitancy_rate.get('measures'):
                        insights.append(f"• **Measures**: {', '.join(hesitancy_rate['measures'][:2])}")
                else:
                    insights.append("\n**Vaccine Hesitancy Data:**")
                    insights.append("• **Data not available (N/A)** for hesitancy measures")
                
                # Report other categories if available
                if neutral_rate.get('data_available'):
                    insights.append(f"\n**Neutral/Uncertain**: {neutral_rate['rate']}%")
                if waiting_rate.get('data_available'):
                    insights.append(f"**Waiting for vaccination**: {waiting_rate['rate']}%")
                
                # Add interpretation
                insights.append("\n**What This Means:**")
                if acceptance_rate.get('data_available') and not hesitancy_rate.get('data_available'):
                    insights.append(f"The data shows **high vaccine acceptance** ({acceptance_rate['rate']}%) among {group_name} individuals in this dataset. ")
                    insights.append("No specific hesitancy data is available, suggesting this group has positive vaccine attitudes.")
                elif acceptance_rate.get('data_available') and hesitancy_rate.get('data_available'):
                    acc_rate = acceptance_rate['rate']
                    hes_rate = hesitancy_rate['rate']
                    if acc_rate > hes_rate:
                        insights.append(f"The data shows **higher acceptance** ({acc_rate}%) than hesitancy ({hes_rate}%) among {group_name} individuals.")
                    else:
                        insights.append(f"The data shows **higher hesitancy** ({hes_rate}%) than acceptance ({acc_rate}%) among {group_name} individuals.")
                
                # Add data quality context
                insights.append(f"\n**Data Quality**: {data_quality.get('measures_with_data', 0)} out of {data_quality.get('total_measures', 0)} measures had valid data.")
                insights.append(f"Categorization confidence: {data_quality.get('categorization_confidence', 0)*100:.0f}%")
                
                return "\n".join(insights)
            else:
                return f"I found data related to {dimension_display}, but couldn't analyze the patterns. The data may need different analysis approaches."
        
        # Generic fallback
        return f"I found {len(data)} records related to your query about vaccine hesitancy. The data shows various patterns that may be relevant to public health decision-making. For more detailed analysis, please try rephrasing your question or asking about specific aspects of the data."
    
    def add_context_explanations(self, source: str, data: List[Dict], analysis: Dict) -> str:
        """Add helpful explanations for technical concepts and data quality"""
        
        context_parts = []
        
        # Add confidence interval explanation if present
        if any('lcl' in str(row) and 'ucl' in str(row) for row in data):
            context_parts.append(
                "\n**Understanding Confidence Intervals:** The data includes confidence intervals (lcl/ucl) "
                "which show the range where we're 95% confident the true percentage falls. "
                "This indicates the reliability of our estimates."
            )
        
        # Add sample size context
        sample_sizes = [row.get('sample_size') for row in data if row.get('sample_size')]
        if sample_sizes:
            avg_sample = sum(float(s) for s in sample_sizes if s) / len([s for s in sample_sizes if s])
            context_parts.append(
                f"\n**Data Quality:** This analysis is based on survey data with an average sample size of "
                f"{avg_sample:.0f} respondents. Larger samples provide more reliable estimates."
            )
        
        # Add data period context
        periods = [row.get('period') for row in data if row.get('period')]
        if periods:
            unique_periods = list(set(str(p) for p in periods if p))
            if unique_periods:
                context_parts.append(
                    f"\n**Data Period:** This analysis covers {', '.join(unique_periods[:2])}{'...' if len(unique_periods) > 2 else ''}."
                )
        
        # Add source-specific context
        if source == "ONS":
            context_parts.append(
                "\n**About ONS Data:** This data comes from the UK Office for National Statistics, "
                "providing nationally representative insights into vaccine hesitancy patterns."
            )
        elif source == "Twitter":
            context_parts.append(
                "\n**About Social Media Data:** This analysis reflects vaccine-related discussions "
                "on social media platforms and may not represent the general population."
            )
        
        return "".join(context_parts)
    
    async def format_response(self, routing_result: Dict, tool_results: List[Dict], ctx: Context) -> str:
        """Format raw SQL results into intelligent, conversational insights"""
        
        try:
            if routing_result.get("routing") == "unavailable":
                return f"""I apologize, but I cannot answer your query with the available data.

**Available Data Sources:**
- **ONS Data**: Demographics (age, sex, ethnicity, religion), employment status, health conditions, vaccine hesitancy trends and barriers
- **Twitter Data**: Sentiment analysis, timeline trends, engagement metrics, influencer analysis, viral content, user behavior

**Sample Queries:**
- "vaccine hesitancy by gender" (ONS)
- "vaccine sentiment over time" (Twitter)
- "top vaccine influencers" (Twitter)
- "vaccine hesitancy in North West region" (ONS)
- "most viral vaccine tweets" (Twitter)
- "vaccine sentiment breakdown" (Twitter)

Please try asking about specific demographics, vaccine hesitancy trends, or social media analysis related to vaccines."""

            if not tool_results or all(result.get("error") for result in tool_results):
                errors = [result.get("error", "Unknown error") for result in tool_results if result.get("error")]
                return f"""I encountered an error while retrieving the data: {'; '.join(errors)}

Please try rephrasing your question or ask about specific vaccine hesitancy topics."""

            # Process successful results with INTELLIGENT ANALYSIS
            response_parts = []
            
            for result in tool_results:
                if result.get("success") and result.get("data"):
                    source = result.get("source", "Unknown")
                    data = result["data"]
                    sql_query = result.get("sql_query", "")
                    user_query = routing_result.get("refined_query", "your query")
                    
                    if data and isinstance(data, list) and len(data) > 0:
                        try:
                            # Enhanced debugging and data inspection
                            ctx.logger.info(f"🔍 [ANALYSIS] Processing {len(data)} rows from {source}")
                            ctx.logger.info(f"🔍 [ANALYSIS] Data columns: {list(data[0].keys()) if data else 'No data'}")
                            ctx.logger.info(f"🔍 [ANALYSIS] First row sample: {data[0] if data else 'No data'}")
                            ctx.logger.info(f"🔍 [ANALYSIS] User query: {user_query}")
                            
                            # STEP 1: Analyze the data for patterns and insights - completely dynamic
                            analysis = None
                            
                            if source == "ONS":
                                # Auto-detect analysis type based on data structure
                                first_row = data[0] if data else {}
                                
                                # Check if this looks like hesitancy reasons data
                                if (any(keyword in str(sql_query).lower() for keyword in ['reasons', 'barriers']) or
                                    ('measure' in first_row and 'percent' in first_row and 
                                     any(keyword in str(first_row.get('measure', '')).lower() for keyword in ['worried', 'concerned', 'side']))):
                                    
                                    ctx.logger.info("🔍 [ANALYSIS] Detected hesitancy reasons data")
                                    analysis = await self.analyze_hesitancy_reasons(data)
                                else:
                                    ctx.logger.info("🔍 [ANALYSIS] Detected demographic patterns data")
                                    analysis = await self.analyze_demographic_patterns(data)
                            
                            elif source == "Twitter":
                                ctx.logger.info("🔍 [ANALYSIS] Analyzing Twitter patterns")
                                analysis = await self.analyze_twitter_patterns(data)
                            
                            else:
                                ctx.logger.info("🔍 [ANALYSIS] Using general analysis")
                                analysis = {'type': 'general', 'data': data}
                            
                            ctx.logger.info(f"🔍 [ANALYSIS] Analysis result type: {analysis.get('type')}")
                            ctx.logger.info(f"🔍 [ANALYSIS] Analysis details: {str(analysis)[:500]}...")
                            
                            # STEP 2: Generate conversational insights using ASI1
                            insights = await self.generate_insights_with_asi1(data, analysis, user_query, source)
                            response_parts.append(insights)
                            
                            # STEP 3: Add context and educational information
                            context = self.add_context_explanations(source, data, analysis)
                            if context.strip():
                                response_parts.append(context)
                            
                            # STEP 4: Add transparency (optional - can be removed for cleaner output)
                            response_parts.append(f"\n*Query executed: {sql_query}*")
                            
                        except Exception as analysis_error:
                            ctx.logger.error(f"Analysis error: {analysis_error}")
                            # Fallback to simpler formatting
                            response_parts.append(f"**{source} Data Analysis:**")
                            response_parts.append(f"Found {len(data)} records related to your query.")
                            if len(data) <= 5:
                                for i, row in enumerate(data, 1):
                                    formatted_row = self._format_row(row, source)
                                    response_parts.append(f"{i}. {formatted_row}")
                    
                    else:
                        response_parts.append(f"No {source.lower()} data found for your query.")
            
            if not response_parts:
                return "No data was found for your query. Please try a different search or check the available data sources."
            
            return "\n".join(response_parts)
            
        except Exception as e:
            ctx.logger.error(f"Response formatting error: {str(e)}")
            return f"I encountered an error processing your query. Please try rephrasing your question or ask about specific vaccine hesitancy topics."

    def _should_show_all_records(self, source: str, data: List[Dict]) -> bool:
        """Determine if we should show all records based on data type"""
        # Timeline data should usually show all records since it's a sequence
        if any(keyword in source.lower() for keyword in ["timeline", "trend", "over time"]):
            return len(data) <= 30  # Show all timeline data up to 30 records
        
        # Column information should show all
        if any(keyword in data[0].keys() if data else [] for keyword in ["column_name", "table_name"]):
            return len(data) <= 20
        
        # Sentiment distribution should show all (usually only 3-4 records)
        if any(keyword in source.lower() for keyword in ["distribution", "breakdown"]):
            return True
        
        # For other Twitter data, be more generous with display
        if source == "Twitter":
            return len(data) <= 20
        
        return len(data) <= 15

    def _get_display_limit(self, source: str, total_records: int) -> int:
        """Get appropriate display limit based on source type"""
        if source == "Twitter":
            return 20  # Twitter data can show more records
        elif "ONS" in source:
            return 15  # ONS data moderate limit
        else:
            return 10  # Default limit

    def _format_row(self, row: Dict, source: str) -> str:
        """Format individual rows based on source type for better readability"""
        
        # Special formatting for timeline data
        if any(key in row for key in ["week_start", "date", "time_period", "day", "hour"]):
            time_key = next((k for k in ["week_start", "date", "time_period", "day", "hour"] if k in row), None)
            if time_key:
                time_val = str(row[time_key])[:19] if row[time_key] else "Unknown"  # Trim timestamp
                other_items = [f"{k}: {v}" for k, v in row.items() if k != time_key]
                return f"📅 {time_val} → {', '.join(other_items)}"
        
        # Special formatting for tweet content
        if "content" in row and "author_username" in row:
            content_preview = str(row["content"])[:100] + "..." if len(str(row["content"])) > 100 else str(row["content"])
            other_items = [f"{k}: {v}" for k, v in row.items() if k not in ["content"]]
            return f'💬 "{content_preview}" | {", ".join(other_items)}'
        
        # Special formatting for user/author data
        if "author_username" in row:
            username = row["author_username"]
            other_items = [f"{k}: {v}" for k, v in row.items() if k != "author_username"]
            return f"👤 @{username} → {', '.join(other_items)}"
        
        # Special formatting for sentiment data
        if "sentiment" in row and "percentage" in row:
            sentiment = str(row["sentiment"]).title()
            percentage = row["percentage"]
            other_items = [f"{k}: {v}" for k, v in row.items() if k not in ["sentiment", "percentage"]]
            emoji = "😊" if sentiment == "Positive" else "😔" if sentiment == "Negative" else "😐"
            return f"{emoji} {sentiment} ({percentage}%) → {', '.join(other_items)}"
        
        # Default formatting
        return ", ".join([f"{k}: {v}" for k, v in row.items()])

    async def process_query(self, user_query: str, ctx: Context) -> str:
        """Main processing pipeline"""
        try:
            # Step 1: Check and route query
            routing_result = await self.check_and_route_query(user_query, ctx)
            ctx.logger.info(f"Routing result: {routing_result}")
            
            routing = routing_result.get("routing", "unavailable")
            refined_query = routing_result.get("refined_query", user_query)
            
            # Step 2: Call appropriate tools
            tool_results = []
            
            if routing == "ons_only":
                ons_result = await self.query_ons_data(refined_query, ctx)
                tool_results.append(ons_result)
                
            elif routing == "twitter_only":
                twitter_result = await self.query_twitter_data(refined_query, ctx)
                tool_results.append(twitter_result)
                
            elif routing == "both":
                ons_result = await self.query_ons_data(refined_query, ctx)
                twitter_result = await self.query_twitter_data(refined_query, ctx)
                tool_results.extend([ons_result, twitter_result])
            
            # Step 3: Format response
            final_response = await self.format_response(routing_result, tool_results, ctx)
            return final_response
            
        except Exception as e:
            ctx.logger.error(f"Query processing error: {str(e)}")
            return f"I encountered an error processing your query: {str(e)}"

# Set up chat protocol and agent
chat_proto = Protocol(spec=chat_protocol_spec)
agent = Agent(
    name='vaccine_hesitancy_insights_agent',
    seed="vaccine_hesitancy_insights_agent_2024",
    port=8003,
    endpoint=["http://vh-insights-chat-agent:8003/submit"],  # Use Docker service name
    mailbox=True
)

client = SupabaseDirectClient()
vh_agent = VaccineHesitancyAgent(client)

@agent.on_event("startup")
async def startup_function(ctx: Context):
    ctx.logger.info("🚀 Starting Vaccine Hesitancy Insights Agent")
    ctx.logger.info("📊 Capabilities: ONS demographic data + Twitter sentiment analysis")
    try:
        await client.connect(ctx)
        ctx.logger.info("✅ Connected to Supabase Postgres directly via asyncpg")
    except Exception as e:
        ctx.logger.error(f"❌ Failed to connect to Postgres: {str(e)}")
        raise

@chat_proto.on_message(model=ChatMessage)
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    try:
        ctx.logger.info(f"📨 Received message from {sender}")
        
        # Check if already processed
        processed_key = f"processed_{msg.msg_id}"
        if ctx.storage.get(processed_key):
            ctx.logger.info(f"Message {msg.msg_id} already processed, skipping")
            return
        
        ctx.storage.set(processed_key, True)
        
        # Send acknowledgement with better error handling
        try:
            ack = ChatAcknowledgement(
                timestamp=datetime.now(timezone.utc),
                acknowledged_msg_id=msg.msg_id
            )
            await ctx.send(sender, ack)
            ctx.logger.info(f"✅ Sent acknowledgement for message {msg.msg_id}")
        except Exception as ack_error:
            ctx.logger.warning(f"⚠️ Acknowledgement failed: {ack_error}, continuing with response...")
            # Continue processing even if acknowledgement fails
        
        # Extract user message
        user_message = next((item.text for item in msg.content if isinstance(item, TextContent)), None)
        if not user_message:
            ctx.logger.warning("No text content found in message")
            return
        
        ctx.logger.info(f"🔍 Processing query: {user_message}")
        
        # Process query through the main agent
        response_text = await vh_agent.process_query(user_message, ctx)
        
        # Send response with enhanced error handling
        try:
            response_msg = ChatMessage(
                timestamp=datetime.now(timezone.utc),
                msg_id=uuid4(),
                content=[TextContent(type="text", text=response_text)]
            )
            
            await ctx.send(sender, response_msg)
            ctx.logger.info(f"✅ Sent response message {response_msg.msg_id}")
            ctx.logger.info(f"📊 Response content preview: {response_text[:200]}...")
            
        except Exception as response_error:
            ctx.logger.error(f"❌ Failed to send response: {response_error}")
            # Try to send a simple error message
            try:
                simple_response = ChatMessage(
                    timestamp=datetime.now(timezone.utc),
                    msg_id=uuid4(),
                    content=[TextContent(type="text", text=f"Enhanced Vaccine Hesitancy Analysis:\n\n{response_text}")]
                )
                await ctx.send(sender, simple_response)
                ctx.logger.info("✅ Sent simplified response as fallback")
            except Exception as fallback_error:
                ctx.logger.error(f"❌ Even fallback response failed: {fallback_error}")
        
    except Exception as e:
        ctx.logger.error(f"❌ Error handling chat message: {str(e)}")
        # Try to send error response
        try:
            error_response = ChatMessage(
                timestamp=datetime.now(timezone.utc),
                msg_id=uuid4(),
                content=[TextContent(type="text", text=f"An error occurred while processing your vaccine hesitancy query: {str(e)}")]
            )
            await ctx.send(sender, error_response)
        except Exception as error_send_error:
            ctx.logger.error(f"❌ Failed to send error response: {error_send_error}")

@chat_proto.on_message(model=ChatAcknowledgement)
async def handle_chat_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    try:
        # Handle acknowledgement with better error handling
        if hasattr(msg, 'acknowledged_msg_id'):
            ctx.logger.info(f"📝 Received acknowledgement from {sender} for message {msg.acknowledged_msg_id}")
        else:
            ctx.logger.info(f"📝 Received acknowledgement from {sender} (no msg_id field)")
    except Exception as e:
        ctx.logger.warning(f"⚠️ Error handling acknowledgement: {e}")
        # Don't let acknowledgement errors break the flow

agent.include(chat_proto)

if __name__ == "__main__":
    try:
        print("""
🤖 Vaccine Hesitancy Insights Agent

📊 Capabilities:
   • ONS demographic vaccine hesitancy data
   • Twitter vaccine sentiment analysis  
   • Comparative analysis across data sources
   • Intelligent query routing and refinement

💬 Chat-based interface with mailbox support
🛑 Stop with Ctrl+C
        """)
        agent.run()
    except Exception as e:
        print(f"❌ Error running agent: {str(e)}")
    finally:
        asyncio.run(client.cleanup())
