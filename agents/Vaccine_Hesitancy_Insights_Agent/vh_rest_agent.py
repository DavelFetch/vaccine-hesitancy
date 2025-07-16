"""
Vaccine Hesitancy REST Agent - Generic API Implementation

This agent provides a single generic /vh endpoint that handles all vaccine hesitancy
demographic dimensions through a unified interface with proper validation and security.

Features:
- Single POST /vh endpoint (eliminates 20 separate endpoints)
- Uses existing MCP client infrastructure (Supabase via Smithery.ai)
- Proper SQL escaping and parameterized queries
- Structured response format with pagination
- Comprehensive validation and whitelisting
- Latest wave/period defaults
- 100 rows default limit

Architecture follows expert feedback to eliminate technical debt and improve maintainability.
"""

from dotenv import load_dotenv
from uagents import Agent, Context, Model
from datetime import datetime, UTC
from typing import Dict, Any, List, Optional, Tuple
import mcp
from mcp.client.streamable_http import streamablehttp_client
import json
import base64
import asyncio
from contextlib import AsyncExitStack
import os

# Load environment variables
load_dotenv()

# Validate required environment variables
required_env_vars = {
    "SUPABASE_ACCESS_TOKEN": os.getenv("SUPABASE_ACCESS_TOKEN"),
    "SMITHERY_API_KEY": os.getenv("SMITHERY_API_KEY"),
    "SUPABASE_PROJECT_ID": os.getenv("SUPABASE_PROJECT_ID")
}

# Check if any required variables are missing
missing_vars = [var for var, value in required_env_vars.items() if not value]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# ============================================================================
# MCP CLIENT (REUSING EXISTING INFRASTRUCTURE)
# ============================================================================

class HesitancyInsightsMCPClient:
    """Reuse existing MCP client infrastructure"""
    def __init__(self):
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.config = {
            "accessToken": required_env_vars["SUPABASE_ACCESS_TOKEN"],
            "readOnly": True,
        }
        self.project_id = required_env_vars["SUPABASE_PROJECT_ID"]
        self.max_retries = 3
        self.retry_delay = 1  # seconds

    async def connect(self, ctx: Context):
        """Connect to Supabase MCP server"""
        config_b64 = base64.b64encode(json.dumps(self.config).encode())
        url = f"https://server.smithery.ai/@supabase-community/supabase-mcp/mcp?config={config_b64}&api_key={required_env_vars['SMITHERY_API_KEY']}&profile=dual-barnacle-C2qHG5"

        read_stream, write_stream, _ = await self.exit_stack.enter_async_context(
            streamablehttp_client(url)
        )
        self.session = await self.exit_stack.enter_async_context(
            mcp.ClientSession(read_stream, write_stream)
        )
        await self.session.initialize()
        ctx.logger.info("Connected to Supabase MCP server")

    async def ensure_connection(self, ctx: Context):
        """Ensure we have an active connection, reconnect if needed"""
        if not self.session:
            await self.connect(ctx)
            return

        try:
            # Try a simple operation to check if session is alive
            await self.session.list_tools()
        except Exception as e:
            ctx.logger.warning(f"Session check failed: {str(e)}. Attempting to reconnect...")
            await self.cleanup()  # Clean up old session
            await self.connect(ctx)  # Create new session

    async def cleanup(self):
        """Cleanup resources"""
        await self.exit_stack.aclose()
        self.session = None

    async def execute_sql(self, query: str, ctx: Context):
        """Execute SQL query with retry logic"""
        for attempt in range(self.max_retries):
            try:
                await self.ensure_connection(ctx)
                return await self.session.call_tool("execute_sql", arguments={
                    "project_id": self.project_id,
                    "query": query
                })
            except Exception as e:
                if attempt == self.max_retries - 1:  # Last attempt
                    raise  # Re-raise the last exception
                ctx.logger.warning(f"Query attempt {attempt + 1} failed: {str(e)}. Retrying...")
                await asyncio.sleep(self.retry_delay)

# ============================================================================
# AGENT SETUP
# ============================================================================

agent = Agent(
    name="vaccine_hesitancy_rest_agent",
    seed="vh_rest_agent_seed_2024",
    port=8005,
    endpoint=["http://localhost:8005/submit"],
    mailbox=False
)

client = HesitancyInsightsMCPClient()

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class VHGenericRequest(Model):
    """Generic request model for vaccine hesitancy data queries"""
    dimension: str
    value: str
    filters: Optional[Dict[str, str]] = None

class VHGenericResponse(Model):
    """Generic response model with structured data and metadata"""
    data: List[Dict[str, Any]]
    sql: str

class VHDimensionsResponse(Model):
    """Response model for available dimensions lookup"""
    dimensions: List[str]
    tables: Dict[str, str]

class HealthResponse(Model):
    """Health check response"""
    status: str
    timestamp: str
    mcp_connected: bool

# ============================================================================
# DIMENSION MAPPING AND VALIDATION
# ============================================================================

# Dimension mapping: dimension_key -> (table_name, primary_column)
DIMENSION_MAPPING = {
    "region": ("vaccine_hesitancy_region", "region"),
    "age_band": ("vaccine_hesitancy_age", "age_band"),
    "age_group": ("vaccine_hesitancy_age_group", "age_group"),
    "sex": ("vaccine_hesitancy_sex", "sex"),
    "ethnicity": ("vaccine_hesitancy_ethnicity", "ethnicity"),
    "religion": ("vaccine_hesitancy_religion", "religion"),
    "disability_status": ("vaccine_hesitancy_disability", "disability_status"),
    "cev_status": ("vaccine_hesitancy_cev", "cev_status"),
    "health_condition": ("vaccine_hesitancy_health_condition", "health_condition"),
    "health_general_condition": ("vaccine_hesitancy_health_general_condition", "health_general_condition"),
    "imd_quintile": ("vaccine_hesitancy_imd_quintile", "imd_quintile"),
    "employment_status": ("vaccine_hesitancy_employment", "employment_status"),
    "expense_affordability": ("vaccine_hesitancy_expense_affordability", "expense_affordability"),
    "household_type": ("vaccine_hesitancy_household_type", "household_type"),
    "caregiver_status": ("vaccine_hesitancy_caregiver_status", "caregiver_status"),
    "age_sex": ("vaccine_hesitancy_age_sex", "group"),
    "trends": ("vaccine_hesitancy_trends", "period"),
    "barriers": ("vaccine_hesitancy_barriers", "block"),
    "reasons": ("vaccine_hesitancy_reasons", "block"),
}

# Allowed values for specific dimensions (updated from user's corrections)
ALLOWED_VALUES = {
    "region": [
        "All adults",
        "East Midlands",
        "East of England",
        "England",
        "London",
        "North East",
        "North West",
        "Scotland",
        "South East",
        "South West",
        "Wales",
        "West Midlands",
        "Yorkshire and The Humber",
    ],
    "age_band": [
        "Aged 16 to 29",
        "Aged 30 to 49",
        "Aged 50 and over",
        "All adults",
    ],
    "age_group": [
        "Aged 16 to 171",
        "Aged 18 to 21",
        "Aged 22 to 25",
        "Aged 26 to 29",
        "Aged 30 to 39",
        "Aged 40 to 49",
        "Aged 50 to 59",
        "Aged 60 to 69",
        "Aged 70 and above",
        "All adults",
    ],
    "sex": [
        "All adults",
        "Men",
        "Women",
    ],
    "ethnicity": [
        "All adults",
        "Black or Black British",
        "Mixed or multiple ethnic groups",
        "Other ethnic group",
        "White",
        "White - British",
        "White - Other White background",
    ],
    "religion": [
        "All adults",
        "Buddhist",
        "Christian",
        "Hindu",
        "Jewish",
        "Muslim",
        "No religion",
        "Other religion",
        "Prefer not to say",
        "Sikh",
    ],
    "disability_status": [
        "All adults",
        "Disabled",
        "Not disabled",
        "Prefer not to say",
    ],
    "cev_status": [
        "All adults",
        "CEV",
        "Not CEV",
    ],
    "health_condition": [
        "All adults",
        "Health condition",
        "No health condition",
        "Don't know",
        "Prefer not to say",
    ],
    "health_general_condition": [
        "All adults",
        "Good/Very good health",
        "Fair health",
        "Bad/Very bad health",
        "Don't know",
        "Prefer not to say"
    ],
    "imd_quintile": [
        "All adults in England",
        "Most deprived",
        "2nd quintile",
        "3rd quintile",
        "4th quintile",
        "Least deprived",
    ],
    "employment_status": [
        "All adults",
        "Employed / self-employed",
        "In employment1",
        "Unemployed",
        "Unpaid family worker",
        "Economically inactive - retired",
        "Economically inactive - other",
    ],
    "expense_affordability": [
        "Able to afford an unexpected, but necessary, expense of £850",
        "All adults",
        "Don't know/Prefer not to say",
        "Unable to afford an unexpected, but necessary, expense of £850",
    ],
    "household_type": [
        "All adults",
        "One adult living alone",
        "Three or more people",
        "Two person household",
    ],
    "caregiver_status": [
        "All adults",
        "Cares for someone in their own home1",
        "Doesn't care for someone in their own home1",
        "Don't know/Prefer not to say",
    ],
    "age_sex": [
        "16 to 29",
        "30 to 49",
        "Aged 50 and over",
        "All adults",
    ],
    "trends": [                       
        "10 December to 10 January5",    
        "13 January to 7 February",   
        "17 February to 14 March",
        "31 March to 25 April",
        "28 April to 23 May",
        "26 May to 20 June",
        "23 June to 18 July",
    ],
    "barriers": [
        "Among those who have received at least one dose of a vaccine",
        "Among those who have not yet received a vaccine",
    ],
    "reasons": [
        "Health2",
        "Catching the coronavirus (COVID-19)2",
        "Fertility2",
        "General hesitation about the vaccine and its safety2",
        "Not needed (now or ever)2",
        "Travel and 'other' reasons2"
    ]

}

# Valid column names for filtering
VALID_FILTER_COLUMNS = {
    "wave_date", "period", "measure", "percent", "lcl", "ucl", "weighted_count", 
    "sample_size", "group", "subgroup", "block", "value_type", "value",
    # Add dimension-specific columns
    "region", "age_band", "age_group", "sex", "ethnicity", "religion",
    "disability_status", "cev_status", "health_condition", "health_general_condition",
    "imd_quintile", "employment_status", "expense_affordability", "household_type",
    "caregiver_status"
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def sql_escape(v: str) -> str:
    """Escape single quotes in SQL string values"""
    return v.replace("'", "''")

def validate_dimension(dimension: str) -> bool:
    """Validate that dimension is in allowed list"""
    return dimension in DIMENSION_MAPPING

def validate_dimension_value(dimension: str, value: str) -> bool:
    """Validate that value is allowed for the given dimension"""
    if dimension not in ALLOWED_VALUES:
        return False
    
    # Special case: allow "all_regions" for region dimension
    if dimension == "region" and value.lower() == "all_regions":
        return True
    
    return value in ALLOWED_VALUES[dimension]

def validate_filter_columns(filters: Optional[Dict[str, str]]) -> bool:
    """Validate filter column names are allowed"""
    if not filters:
        return True
    return all(col in VALID_FILTER_COLUMNS for col in filters.keys())

def build_vh_query(
    dimension: str, 
    value: str, 
    filters: Optional[Dict[str, str]] = None
) -> str:
    """
    Build SQL query for vaccine hesitancy data
    
    Returns:
        Simple data query without pagination
    """
    table_name, primary_column = DIMENSION_MAPPING[dimension]
    
    # Special case: return all regions for map visualization
    if dimension == "region" and value.lower() == "all_regions":
        # Get all individual regions (excluding aggregates)
        individual_regions = [r for r in ALLOWED_VALUES["region"] if r not in ["England", "All adults"]]
        # Properly escape each region name for IN clause
        escaped_regions = [f"'{sql_escape(r)}'" for r in individual_regions]
        region_list = ", ".join(escaped_regions)
        where_conditions = [f"{primary_column} IN ({region_list})"]
    else:
        # Escape the primary value
        escaped_value = sql_escape(value)
        # Base WHERE clause
        if primary_column == "group":
            where_conditions = [f'"group" = \'{escaped_value}\'']
        else:
            where_conditions = [f"{primary_column} = '{escaped_value}'"]
    
    # Add filters if provided
    if filters:
        for col, val in filters.items():
            escaped_val = sql_escape(val)
            col_escaped = f'"{col}"' if col in ["group", "subgroup"] else col
            where_conditions.append(f"{col_escaped} = '{escaped_val}'")
    
    # Add latest wave/period default if not specified
    if filters is None or ("wave_date" not in filters and "period" not in filters):
        # Check if table has wave_date or period column
        if table_name in ["vaccine_hesitancy_age", "vaccine_hesitancy_age_group", 
                         "vaccine_hesitancy_sex", "vaccine_hesitancy_ethnicity", "vaccine_hesitancy_religion",
                         "vaccine_hesitancy_disability", "vaccine_hesitancy_cev", "vaccine_hesitancy_health_condition",
                         "vaccine_hesitancy_health_general_condition", "vaccine_hesitancy_imd_quintile",
                         "vaccine_hesitancy_employment", "vaccine_hesitancy_expense_affordability",
                         "vaccine_hesitancy_household_type", "vaccine_hesitancy_caregiver_status", "vaccine_hesitancy_age_sex"]:
            where_conditions.append(f"wave_date = (SELECT MAX(wave_date) FROM {table_name})")
        elif table_name in ["vaccine_hesitancy_trends", "vaccine_hesitancy_reasons"]:
            # Only add this if value is not provided
            if not value:
                where_conditions.append(f"period = (SELECT MAX(period) FROM {table_name})")
        # Note: vaccine_hesitancy_region and vaccine_hesitancy_barriers have no time dimension
    
    where_clause = " AND ".join(where_conditions)
    
    # Simple data query without pagination bullshit
    data_query = f"SELECT * FROM {table_name} WHERE {where_clause}"
    
    return data_query

def parse_mcp_result(result) -> List[Dict[str, Any]]:
    """Parse MCP result into structured data"""
    try:
        # Handle the result content similar to existing endpoints
        if not result or not result.content:
            return []
        
        content = result.content[0].text if isinstance(result.content, list) else result.content
        data = json.loads(content)
        
        # Ensure data is a list
        if isinstance(data, dict):
            return [data]
        elif isinstance(data, list):
            return data
        else:
            return []
            
    except Exception:
        return []

# ============================================================================
# REST ENDPOINTS
# ============================================================================

@agent.on_rest_post("/vh", VHGenericRequest, VHGenericResponse)
async def handle_vh_query(ctx: Context, request: VHGenericRequest) -> VHGenericResponse:
    """
    Generic vaccine hesitancy data endpoint
    
    Handles all demographic dimensions through a single unified interface
    with proper validation and escaping.
    """
    try:
        # Debug logging
        ctx.logger.info(f"Received request: dimension={request.dimension}, value={request.value}")
        ctx.logger.info(f"Raw filters type: {type(getattr(request, 'filters', None))}")
        
        # Validate dimension
        if not validate_dimension(request.dimension):
            raise ValueError(f"Invalid dimension: {request.dimension}")
        
        # Validate dimension value
        if not validate_dimension_value(request.dimension, request.value):
            raise ValueError(f"Invalid value '{request.value}' for dimension '{request.dimension}'")
        
        # Validate filters if provided - with defensive type checking
        filters = getattr(request, 'filters', None)
        if filters is not None and isinstance(filters, dict) and not validate_filter_columns(filters):
            invalid_keys = set(filters.keys()) - VALID_FILTER_COLUMNS
            raise ValueError(f"Invalid filter columns: {invalid_keys}")
        elif filters is not None and not isinstance(filters, dict):
            # Handle case where filters is not a dict (e.g., FieldInfo object)
            ctx.logger.warning(f"Invalid filters type: {type(filters)}, treating as None")
            filters = None
        
        # Build query
        data_query = build_vh_query(
            request.dimension,
            request.value,
            filters
        )
        
        # Execute data query
        data_result = await client.execute_sql(data_query, ctx)
        data = parse_mcp_result(data_result)
        
        ctx.logger.info(
            f"VH query successful: dimension={request.dimension}, "
            f"value={request.value}, returned={len(data)} rows"
        )
        
        return VHGenericResponse(
            data=data,
            sql=data_query
        )
        
    except ValueError as e:
        ctx.logger.warning(f"Validation error in VH query: {e}")
        raise
    except Exception as e:
        ctx.logger.error(f"Unexpected error in VH query: {e}")
        raise

@agent.on_rest_get("/dimensions", VHDimensionsResponse)
async def get_dimensions(ctx: Context) -> VHDimensionsResponse:
    """Get available dimensions and their table mappings"""
    try:
        dimensions = list(DIMENSION_MAPPING.keys())
        tables = {dim: table for dim, (table, _) in DIMENSION_MAPPING.items()}
        
        ctx.logger.info(f"Dimensions lookup: {len(dimensions)} dimensions available")
        
        return VHDimensionsResponse(
            dimensions=dimensions,
            tables=tables
        )
        
    except Exception as e:
        ctx.logger.error(f"Error in dimensions lookup: {e}")
        raise

@agent.on_rest_get("/health", HealthResponse)
async def health_check(ctx: Context) -> HealthResponse:
    """Health check endpoint"""
    try:
        # Test MCP connection
        mcp_connected = False
        try:
            await client.ensure_connection(ctx)
            await client.session.list_tools()
            mcp_connected = True
        except Exception as e:
            ctx.logger.warning(f"MCP connection test failed: {e}")
        
        ctx.logger.info(f"Health check: mcp_connected={mcp_connected}")
        
        return HealthResponse(
            status="healthy" if mcp_connected else "degraded",
            timestamp=datetime.now(UTC).isoformat(),
            mcp_connected=mcp_connected
        )
        
    except Exception as e:
        ctx.logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(UTC).isoformat(),
            mcp_connected=False
        )

# ============================================================================
# AGENT LIFECYCLE
# ============================================================================

@agent.on_event("startup")
async def startup(ctx: Context):
    """Agent startup handler"""
    ctx.logger.info("🚀 Vaccine Hesitancy REST Agent starting...")
    ctx.logger.info(f"📍 Agent address: {agent.address}")
    ctx.logger.info(f"🌐 REST API available at: http://localhost:8005")
    
    # Connect to MCP server
    try:
        await client.connect(ctx)
        ctx.logger.info("✅ Connected to Supabase MCP server")
    except Exception as e:
        ctx.logger.error(f"❌ MCP connection failed: {e}")
    
    ctx.logger.info("📋 Available endpoints:")
    ctx.logger.info("   • POST /vh        - Generic vaccine hesitancy data query")
    ctx.logger.info("   • GET  /dimensions - Available dimensions lookup")
    ctx.logger.info("   • GET  /health     - Health check")
    
    ctx.logger.info("🔗 Example usage:")
    ctx.logger.info("   curl -X POST http://localhost:8005/vh \\")
    ctx.logger.info("        -H 'Content-Type: application/json' \\")
    ctx.logger.info("        -d '{\"dimension\": \"region\", \"value\": \"England\"}'")

@agent.on_event("shutdown")
async def shutdown(ctx: Context):
    """Agent shutdown handler"""
    ctx.logger.info("🛑 Vaccine Hesitancy REST Agent shutting down...")
    await client.cleanup()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
🤖 Vaccine Hesitancy REST Agent

📊 Single Generic API for All Demographic Dimensions

Features:
• Single POST /vh endpoint (eliminates 20 separate endpoints)
• Uses existing MCP infrastructure (Supabase via Smithery.ai)
• Proper SQL escaping and validation
• Structured response format with pagination
• 100 rows default limit
• Latest wave/period defaults

📋 Endpoints:
   • POST /vh        - Generic vaccine hesitancy data query
   • GET  /dimensions - Available dimensions lookup  
   • GET  /health     - Health check

🔗 Example:
   curl -X POST http://localhost:8005/vh \\
        -H 'Content-Type: application/json' \\
        -d '{"dimension": "region", "value": "England", "limit": 50}'

🛑 Stop with Ctrl+C
    """)
    agent.run() 