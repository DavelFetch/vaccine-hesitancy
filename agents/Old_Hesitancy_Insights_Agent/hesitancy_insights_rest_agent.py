from dotenv import load_dotenv
from uagents_core.contrib.protocols.chat import (
    chat_protocol_spec,
    ChatMessage,
    ChatAcknowledgement,
    TextContent,
    StartSessionContent,
)
from uagents import Agent, Context, Protocol
from uagents.setup import fund_agent_if_low
from datetime import datetime, timezone, timedelta
from uuid import uuid4
import mcp
from mcp.client.streamable_http import streamablehttp_client
import json
import base64
import asyncio
from typing import Dict, List, Optional, Any
from contextlib import AsyncExitStack
import os
from vh_models import (
    VHRegionRequest, VHRegionResponse,
    VHSexRequest, VHSexResponse,
    VHAgeBandRequest, VHAgeBandResponse,
    VHAgeGroupRequest, VHAgeGroupResponse,
    VHEthnicityRequest, VHEthnicityResponse,
    VHReligionRequest, VHReligionResponse,
    VHDisabilityRequest, VHDisabilityResponse,
    VHCEVRequest, VHCEVResponse,
    VHUnderlyingHealthRequest, VHUnderlyingHealthResponse,
    VHHealthGeneralRequest, VHHealthGeneralResponse,
    VHIMDRequest, VHIMDResponse,
    VHEmploymentRequest, VHEmploymentResponse,
    VHExpenseRequest, VHExpenseResponse,
    VHHouseholdRequest, VHHouseholdResponse,
    VHCaregiverRequest, VHCaregiverResponse,
    VHAgeSexRequest, VHAgeSexResponse,
    VHTrendsRequest, VHTrendsResponse,
    VHBarriersRequest, VHBarriersResponse,
    VHReasonsRequest, VHReasonsResponse,
    VHAgentRequest, VHAgentResponse
)

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

class HesitancyInsightsMCPClient:
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
    
agent = Agent(
    name="hesitancy_insights_rest_agent",
    port=8000,
    seed="hesitancy_insights_rest_agent",
    mailbox=True
)
client = HesitancyInsightsMCPClient()

@agent.on_event("startup")
async def startup_function(ctx: Context):
    ctx.logger.info("Starting up Hesitancy Insights Rest Agent")
    await client.connect(ctx)

@agent.on_rest_post("/vh_region", VHRegionRequest, VHRegionResponse)
async def handle_vh_region(ctx: Context, msg: VHRegionRequest) -> VHRegionResponse:
    try:
        ctx.logger.info(f"Received vaccination hesitancy data based on region: {msg.region}")
        
        # Use the region from the request in the query
        query = f"""
            SELECT * 
            FROM vaccine_hesitancy_region 
            WHERE region = '{msg.region}'
            LIMIT 10;
        """
        
        result = await client.execute_sql(query, ctx)
        ctx.logger.info(f"Query executed successfully")
        
        # Check if we got any results
        if not result or not result.content:
            return VHRegionResponse(
                response="No data found for the specified region",
                status="not_found"
            )
        
        # Convert the result to a string format
        try:
            # Try to parse the content as JSON
            content = result.content[0].text if isinstance(result.content, list) else result.content
            data = json.loads(content)
            formatted_response = json.dumps(data, indent=2)
        except Exception as e:
            # If JSON parsing fails, use the raw string
            formatted_response = str(result.content)
            
        return VHRegionResponse(
            response=formatted_response,
            status="success"
        )
        
    except Exception as e:
        ctx.logger.error(f"Error processing query: {str(e)}")
        return VHRegionResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_post("/vh_sex", VHSexRequest, VHSexResponse)
async def handle_vh_sex(ctx: Context, msg: VHSexRequest) -> VHSexResponse:
    try:
        ctx.logger.info(f"Received vaccination hesitancy data based on sex: {msg.sex}")
        
        # Use the sex from the request in the query
        query = f"""
            SELECT * 
            FROM vaccine_hesitancy_sex 
            WHERE sex = '{msg.sex}'
            LIMIT 10;
        """
        
        result = await client.execute_sql(query, ctx)
        ctx.logger.info(f"Query executed successfully")
        
        # Check if we got any results
        if not result or not result.content:
            return VHSexResponse(
                response="No data found for the specified sex",
                status="not_found"
            )
        
        # Convert the result to a string format
        try:
            # Try to parse the content as JSON
            content = result.content[0].text if isinstance(result.content, list) else result.content
            data = json.loads(content)
            formatted_response = json.dumps(data, indent=2)
        except Exception as e:
            # If JSON parsing fails, use the raw string
            formatted_response = str(result.content)
            
        return VHSexResponse(
            response=formatted_response,
            status="success"
        )
        
    except Exception as e:
        ctx.logger.error(f"Error processing query: {str(e)}")
        return VHSexResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_post("/vh_age_band", VHAgeBandRequest, VHAgeBandResponse)
async def handle_vh_age_band(ctx: Context, msg: VHAgeBandRequest) -> VHAgeBandResponse:
    try:
        ctx.logger.info(f"Received vaccination hesitancy data based on age band: {msg.age_band}")
        
        # Use the age band from the request in the query
        query = f"""
            SELECT * 
            FROM vaccine_hesitancy_age 
            WHERE age_band = '{msg.age_band}'
            LIMIT 10;
        """
        
        result = await client.execute_sql(query, ctx)
        ctx.logger.info(f"Query executed successfully")
        
        # Check if we got any results
        if not result or not result.content:
            return VHAgeBandResponse(
                response="No data found for the specified age band",
                status="not_found"
            )
        
        # Convert the result to a string format
        try:
            # Try to parse the content as JSON
            content = result.content[0].text if isinstance(result.content, list) else result.content
            data = json.loads(content)
            formatted_response = json.dumps(data, indent=2)
        except Exception as e:
            # If JSON parsing fails, use the raw string
            formatted_response = str(result.content)
            
        return VHAgeBandResponse(
            response=formatted_response,
            status="success"
        )
        
    except Exception as e:
        ctx.logger.error(f"Error processing query: {str(e)}")
        return VHAgeBandResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_post("/vh_age_group", VHAgeGroupRequest, VHAgeGroupResponse)
async def handle_vh_age_group(ctx: Context, msg: VHAgeGroupRequest) -> VHAgeGroupResponse:
    try:
        ctx.logger.info(f"Received vaccination hesitancy data based on age group: {msg.age_group}")
        
        # Use the age group from the request in the query
        query = f"""
            SELECT * 
            FROM vaccine_hesitancy_age_group 
            WHERE age_group = '{msg.age_group}'
            LIMIT 10;
        """
        
        result = await client.execute_sql(query, ctx)
        ctx.logger.info(f"Query executed successfully")
        
        # Check if we got any results
        if not result or not result.content:
            return VHAgeGroupResponse(
                response="No data found for the specified age group",
                status="not_found"
            )
        
        # Convert the result to a string format
        try:
            # Try to parse the content as JSON
            content = result.content[0].text if isinstance(result.content, list) else result.content
            data = json.loads(content)
            formatted_response = json.dumps(data, indent=2)
        except Exception as e:
            # If JSON parsing fails, use the raw string
            formatted_response = str(result.content)
            
        return VHAgeGroupResponse(
            response=formatted_response,
            status="success"
        )
        
    except Exception as e:
        ctx.logger.error(f"Error processing query: {str(e)}")
        return VHAgeGroupResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_post("/vh_ethnicity", VHEthnicityRequest, VHEthnicityResponse)
async def handle_vh_ethnicity(ctx: Context, msg: VHEthnicityRequest) -> VHEthnicityResponse:
    try:
        ctx.logger.info(f"Received vaccination hesitancy data based on ethnicity: {msg.ethnicity}")
        
        query = f"""
            SELECT * 
            FROM vaccine_hesitancy_ethnicity 
            WHERE ethnicity = '{msg.ethnicity}'
            LIMIT 10;
        """
        
        result = await client.execute_sql(query, ctx)
        ctx.logger.info(f"Query executed successfully")
        
        if not result or not result.content:
            return VHEthnicityResponse(
                response="No data found for the specified ethnicity",
                status="not_found"
            )
        
        try:
            content = result.content[0].text if isinstance(result.content, list) else result.content
            data = json.loads(content)
            formatted_response = json.dumps(data, indent=2)
        except Exception as e:
            formatted_response = str(result.content)
            
        return VHEthnicityResponse(
            response=formatted_response,
            status="success"
        )
        
    except Exception as e:
        ctx.logger.error(f"Error processing query: {str(e)}")
        return VHEthnicityResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_post("/vh_religion", VHReligionRequest, VHReligionResponse)
async def handle_vh_religion(ctx: Context, msg: VHReligionRequest) -> VHReligionResponse:
    try:
        ctx.logger.info(f"Received vaccination hesitancy data based on religion: {msg.religion}")
        
        query = f"""
            SELECT * 
            FROM vaccine_hesitancy_religion 
            WHERE religion = '{msg.religion}'
            LIMIT 10;
        """
        
        result = await client.execute_sql(query, ctx)
        ctx.logger.info(f"Query executed successfully")
        
        if not result or not result.content:
            return VHReligionResponse(
                response="No data found for the specified religion",
                status="not_found"
            )
        
        try:
            content = result.content[0].text if isinstance(result.content, list) else result.content
            data = json.loads(content)
            formatted_response = json.dumps(data, indent=2)
        except Exception as e:
            formatted_response = str(result.content)
            
        return VHReligionResponse(
            response=formatted_response,
            status="success"
        )
        
    except Exception as e:
        ctx.logger.error(f"Error processing query: {str(e)}")
        return VHReligionResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_post("/vh_disability", VHDisabilityRequest, VHDisabilityResponse)
async def handle_vh_disability(ctx: Context, msg: VHDisabilityRequest) -> VHDisabilityResponse:
    try:
        ctx.logger.info(f"Received vaccination hesitancy data based on disability status: {msg.status}")
        
        query = f"""
            SELECT * 
            FROM vaccine_hesitancy_disability 
            WHERE disability_status = '{msg.status}'
            LIMIT 10;
        """
        
        result = await client.execute_sql(query, ctx)
        ctx.logger.info(f"Query executed successfully")
        
        if not result or not result.content:
            return VHDisabilityResponse(
                response="No data found for the specified disability status",
                status="not_found"
            )
        
        try:
            content = result.content[0].text if isinstance(result.content, list) else result.content
            data = json.loads(content)
            formatted_response = json.dumps(data, indent=2)
        except Exception as e:
            formatted_response = str(result.content)
            
        return VHDisabilityResponse(
            response=formatted_response,
            status="success"
        )
        
    except Exception as e:
        ctx.logger.error(f"Error processing query: {str(e)}")
        return VHDisabilityResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_post("/vh_cev", VHCEVRequest, VHCEVResponse)
async def handle_vh_cev(ctx: Context, msg: VHCEVRequest) -> VHCEVResponse:
    try:
        ctx.logger.info(f"Received vaccination hesitancy data based on CEV status: {msg.status}")
        
        query = f"""
            SELECT * 
            FROM vaccine_hesitancy_cev 
            WHERE cev_status = '{msg.status}'
            LIMIT 10;
        """
        
        result = await client.execute_sql(query, ctx)
        ctx.logger.info(f"Query executed successfully")
        
        if not result or not result.content:
            return VHCEVResponse(
                response="No data found for the specified CEV status",
                status="not_found"
            )
        
        try:
            content = result.content[0].text if isinstance(result.content, list) else result.content
            data = json.loads(content)
            formatted_response = json.dumps(data, indent=2)
        except Exception as e:
            formatted_response = str(result.content)
            
        return VHCEVResponse(
            response=formatted_response,
            status="success"
        )
        
    except Exception as e:
        ctx.logger.error(f"Error processing query: {str(e)}")
        return VHCEVResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_post("/vh_health_condition", VHUnderlyingHealthRequest, VHUnderlyingHealthResponse)
async def handle_vh_health_condition(ctx: Context, msg: VHUnderlyingHealthRequest) -> VHUnderlyingHealthResponse:
    try:
        ctx.logger.info(f"Received vaccination hesitancy data based on health condition: {msg.status}")
        
        query = f"""
            SELECT * 
            FROM vaccine_hesitancy_health_condition 
            WHERE health_condition = '{msg.status}'
            LIMIT 10;
        """
        
        result = await client.execute_sql(query, ctx)
        ctx.logger.info(f"Query executed successfully")
        
        if not result or not result.content:
            return VHUnderlyingHealthResponse(
                response="No data found for the specified health condition",
                status="not_found"
            )
        
        try:
            content = result.content[0].text if isinstance(result.content, list) else result.content
            data = json.loads(content)
            formatted_response = json.dumps(data, indent=2)
        except Exception as e:
            formatted_response = str(result.content)
            
        return VHUnderlyingHealthResponse(
            response=formatted_response,
            status="success"
        )
        
    except Exception as e:
        ctx.logger.error(f"Error processing query: {str(e)}")
        return VHUnderlyingHealthResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_post("/vh_health_general", VHHealthGeneralRequest, VHHealthGeneralResponse)
async def handle_vh_health_general(ctx: Context, msg: VHHealthGeneralRequest) -> VHHealthGeneralResponse:
    try:
        ctx.logger.info(f"Received vaccination hesitancy data based on health general condition: {msg.condition}")
        
        query = f"""
            SELECT * 
            FROM vaccine_hesitancy_health_general_condition 
            WHERE health_general_condition = '{msg.condition}'
            LIMIT 10;
        """
        
        result = await client.execute_sql(query, ctx)
        ctx.logger.info(f"Query executed successfully")
        
        if not result or not result.content:
            return VHHealthGeneralResponse(
                response="No data found for the specified health general condition",
                status="not_found"
            )
        
        try:
            content = result.content[0].text if isinstance(result.content, list) else result.content
            data = json.loads(content)
            formatted_response = json.dumps(data, indent=2)
        except Exception as e:
            formatted_response = str(result.content)
            
        return VHHealthGeneralResponse(
            response=formatted_response,
            status="success"
        )
        
    except Exception as e:
        ctx.logger.error(f"Error processing query: {str(e)}")
        return VHHealthGeneralResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_post("/vh_imd", VHIMDRequest, VHIMDResponse)
async def handle_vh_imd(ctx: Context, msg: VHIMDRequest) -> VHIMDResponse:
    try:
        ctx.logger.info(f"Received vaccination hesitancy data based on IMD quintile: {msg.quintile}")
        
        query = f"""
            SELECT * 
            FROM vaccine_hesitancy_imd_quintile 
            WHERE imd_quintile = '{msg.quintile}'
            LIMIT 10;
        """
        
        result = await client.execute_sql(query, ctx)
        ctx.logger.info(f"Query executed successfully")
        
        if not result or not result.content:
            return VHIMDResponse(
                response="No data found for the specified IMD quintile",
                status="not_found"
            )
        
        try:
            content = result.content[0].text if isinstance(result.content, list) else result.content
            data = json.loads(content)
            formatted_response = json.dumps(data, indent=2)
        except Exception as e:
            formatted_response = str(result.content)
            
        return VHIMDResponse(
            response=formatted_response,
            status="success"
        )
        
    except Exception as e:
        ctx.logger.error(f"Error processing query: {str(e)}")
        return VHIMDResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_post("/vh_employment", VHEmploymentRequest, VHEmploymentResponse)
async def handle_vh_employment(ctx: Context, msg: VHEmploymentRequest) -> VHEmploymentResponse:
    try:
        ctx.logger.info(f"Received vaccination hesitancy data based on employment status: {msg.status}")
        
        query = f"""
            SELECT * 
            FROM vaccine_hesitancy_employment 
            WHERE employment_status = '{msg.status}'
            LIMIT 10;
        """
        
        result = await client.execute_sql(query, ctx)
        ctx.logger.info(f"Query executed successfully")
        
        if not result or not result.content:
            return VHEmploymentResponse(
                response="No data found for the specified employment status",
                status="not_found"
            )
        
        try:
            content = result.content[0].text if isinstance(result.content, list) else result.content
            data = json.loads(content)
            formatted_response = json.dumps(data, indent=2)
        except Exception as e:
            formatted_response = str(result.content)
            
        return VHEmploymentResponse(
            response=formatted_response,
            status="success"
        )
        
    except Exception as e:
        ctx.logger.error(f"Error processing query: {str(e)}")
        return VHEmploymentResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_post("/vh_expense", VHExpenseRequest, VHExpenseResponse)
async def handle_vh_expense(ctx: Context, msg: VHExpenseRequest) -> VHExpenseResponse:
    try:
        ctx.logger.info(f"Received vaccination hesitancy data based on expense affordability: {msg.ability}")
        
        query = f"""
            SELECT * 
            FROM vaccine_hesitancy_expense_affordability 
            WHERE expense_affordability = '{msg.ability}'
            LIMIT 10;
        """
        
        result = await client.execute_sql(query, ctx)
        ctx.logger.info(f"Query executed successfully")
        
        if not result or not result.content:
            return VHExpenseResponse(
                response="No data found for the specified expense affordability",
                status="not_found"
            )
        
        try:
            content = result.content[0].text if isinstance(result.content, list) else result.content
            data = json.loads(content)
            formatted_response = json.dumps(data, indent=2)
        except Exception as e:
            formatted_response = str(result.content)
            
        return VHExpenseResponse(
            response=formatted_response,
            status="success"
        )
        
    except Exception as e:
        ctx.logger.error(f"Error processing query: {str(e)}")
        return VHExpenseResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_post("/vh_household", VHHouseholdRequest, VHHouseholdResponse)
async def handle_vh_household(ctx: Context, msg: VHHouseholdRequest) -> VHHouseholdResponse:
    try:
        ctx.logger.info(f"Received vaccination hesitancy data based on household type: {msg.type}")
        
        query = f"""
            SELECT * 
            FROM vaccine_hesitancy_household_type 
            WHERE household_type = '{msg.type}'
            LIMIT 10;
        """
        
        result = await client.execute_sql(query, ctx)
        ctx.logger.info(f"Query executed successfully")
        
        if not result or not result.content:
            return VHHouseholdResponse(
                response="No data found for the specified household type",
                status="not_found"
            )
        
        try:
            content = result.content[0].text if isinstance(result.content, list) else result.content
            data = json.loads(content)
            formatted_response = json.dumps(data, indent=2)
        except Exception as e:
            formatted_response = str(result.content)
            
        return VHHouseholdResponse(
            response=formatted_response,
            status="success"
        )
        
    except Exception as e:
        ctx.logger.error(f"Error processing query: {str(e)}")
        return VHHouseholdResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_post("/vh_caregiver", VHCaregiverRequest, VHCaregiverResponse)
async def handle_vh_caregiver(ctx: Context, msg: VHCaregiverRequest) -> VHCaregiverResponse:
    try:
        ctx.logger.info(f"Received vaccination hesitancy data based on caregiver status: {msg.status}")
        
        query = f"""
            SELECT * 
            FROM vaccine_hesitancy_caregiver_status 
            WHERE caregiver_status = '{msg.status}'
            LIMIT 10;
        """
        
        result = await client.execute_sql(query, ctx)
        ctx.logger.info(f"Query executed successfully")
        
        if not result or not result.content:
            return VHCaregiverResponse(
                response="No data found for the specified caregiver status",
                status="not_found"
            )
        
        try:
            content = result.content[0].text if isinstance(result.content, list) else result.content
            data = json.loads(content)
            formatted_response = json.dumps(data, indent=2)
        except Exception as e:
            formatted_response = str(result.content)
            
        return VHCaregiverResponse(
            response=formatted_response,
            status="success"
        )
        
    except Exception as e:
        ctx.logger.error(f"Error processing query: {str(e)}")
        return VHCaregiverResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_post("/vh_age_sex", VHAgeSexRequest, VHAgeSexResponse)
async def handle_vh_age_sex(ctx: Context, msg: VHAgeSexRequest) -> VHAgeSexResponse:
    try:
        ctx.logger.info(f"Received vaccination hesitancy data by age/sex: group={msg.group}, subgroup={msg.subgroup}, measure={msg.measure}")
        
        # Build WHERE clause
        where_clauses = [f"\"group\" = '{msg.group}'"]
        if msg.subgroup is not None:
            where_clauses.append(f"subgroup = '{msg.subgroup}'")
        if msg.measure is not None:
            where_clauses.append(f"measure = '{msg.measure}'")
        where_sql = " AND ".join(where_clauses)
        
        query = f"""
            SELECT *
            FROM vaccine_hesitancy_age_sex
            WHERE {where_sql}
            LIMIT 10;
        """
        
        result = await client.execute_sql(query, ctx)
        ctx.logger.info(f"Query executed successfully")
        
        if not result or not result.content:
            return VHAgeSexResponse(
                response="No data found for the specified filters",
                status="not_found"
            )
        
        try:
            content = result.content[0].text if isinstance(result.content, list) else result.content
            data = json.loads(content)
            formatted_response = json.dumps(data, indent=2)
        except Exception as e:
            formatted_response = str(result.content)
            
        return VHAgeSexResponse(
            response=formatted_response,
            status="success"
        )
        
    except Exception as e:
        ctx.logger.error(f"Error processing query: {str(e)}")
        return VHAgeSexResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_post("/vh_trends", VHTrendsRequest, VHTrendsResponse)
async def handle_vh_trends(ctx: Context, msg: VHTrendsRequest) -> VHTrendsResponse:
    try:
        ctx.logger.info(f"Received vaccine hesitancy trends data: period={msg.period}, block={msg.block}, measure={msg.measure}, value_type={msg.value_type}")
        
        # Build WHERE clause
        where_clauses = []
        if msg.period is not None:
            where_clauses.append(f"period = '{msg.period}'")
        if msg.block is not None:
            where_clauses.append(f"\"block\" = '{sql_escape(msg.block)}'")
        if msg.measure is not None:
            where_clauses.append(f"\"measure\" = '{sql_escape(msg.measure)}'")
        if msg.value_type is not None:
            where_clauses.append(f"\"value_type\" = '{sql_escape(msg.value_type)}'")
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        query = f"""
            SELECT *
            FROM vaccine_hesitancy_trends
            WHERE {where_sql}
            LIMIT 10;
        """
        
        result = await client.execute_sql(query, ctx)
        ctx.logger.info(f"Query executed successfully")
        
        if not result or not result.content:
            return VHTrendsResponse(
                response="No data found for the specified filters",
                status="not_found"
            )
        
        try:
            content = result.content[0].text if isinstance(result.content, list) else result.content
            data = json.loads(content)
            formatted_response = json.dumps(data, indent=2)
        except Exception as e:
            formatted_response = str(result.content)
            
        return VHTrendsResponse(
            response=formatted_response,
            status="success"
        )
        
    except Exception as e:
        ctx.logger.error(f"Error processing query: {str(e)}")
        return VHTrendsResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_post("/vh_barriers", VHBarriersRequest, VHBarriersResponse)
async def handle_vh_barriers(ctx: Context, msg: VHBarriersRequest) -> VHBarriersResponse:
    try:
        ctx.logger.info(f"Received vaccine hesitancy barriers data: block={msg.block}, group={msg.group}, measure={msg.measure}, value_type={msg.value_type}")
        
        # Build WHERE clause
        where_clauses = []
        if msg.block is not None:
            where_clauses.append(f"\"block\" = '{sql_escape(msg.block)}'")
        if msg.group is not None:
            where_clauses.append(f"\"group\" = '{sql_escape(msg.group)}'")
        if msg.measure is not None:
            where_clauses.append(f"\"measure\" = '{sql_escape(msg.measure)}'")
        if msg.value_type is not None:
            where_clauses.append(f"\"value_type\" = '{sql_escape(msg.value_type)}'")
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        query = f"""
            SELECT *
            FROM vaccine_hesitancy_barriers
            WHERE {where_sql}
            LIMIT 10;
        """
        
        result = await client.execute_sql(query, ctx)
        ctx.logger.info(f"Query executed successfully")
        
        if not result or not result.content:
            return VHBarriersResponse(
                response="No data found for the specified filters",
                status="not_found"
            )
        
        try:
            content = result.content[0].text if isinstance(result.content, list) else result.content
            data = json.loads(content)
            formatted_response = json.dumps(data, indent=2)
        except Exception as e:
            formatted_response = str(result.content)
            
        return VHBarriersResponse(
            response=formatted_response,
            status="success"
        )
        
    except Exception as e:
        ctx.logger.error(f"Error processing query: {str(e)}")
        return VHBarriersResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_post("/vh_reasons", VHReasonsRequest, VHReasonsResponse)
async def handle_vh_reasons(ctx: Context, msg: VHReasonsRequest) -> VHReasonsResponse:
    try:
        ctx.logger.info(f"Received vaccine hesitancy reasons data: period={msg.period}, block={msg.block}, measure={msg.measure}, group={msg.group}")
        # Build WHERE clause
        where_clauses = []
        if msg.period is not None:
            period_escaped = msg.period.replace("'", "''")
            where_clauses.append(f"period = '{period_escaped}'")
        if msg.block is not None:
            block_escaped = msg.block.replace("'", "''")
            where_clauses.append(f'"block" = \'{block_escaped}\'')
        if msg.measure is not None:
            measure_escaped = msg.measure.replace("'", "''")
            where_clauses.append(f'"measure" = \'{measure_escaped}\'')
        if msg.group is not None:
            group_escaped = msg.group.replace("'", "''")
            where_clauses.append(f'"group" = \'{group_escaped}\'')
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        query = f"""
            SELECT *
            FROM vaccine_hesitancy_reasons
            WHERE {where_sql}
            LIMIT 20;
        """
        result = await client.execute_sql(query, ctx)
        ctx.logger.info(f"Query executed successfully")
        if not result or not result.content:
            return VHReasonsResponse(
                response="No data found for the specified filters",
                status="not_found"
            )
        try:
            content = result.content[0].text if isinstance(result.content, list) else result.content
            data = json.loads(content)
            formatted_response = json.dumps(data, indent=2)
        except Exception as e:
            formatted_response = str(result.content)
        return VHReasonsResponse(
            response=formatted_response,
            status="success"
        )
    except Exception as e:
        ctx.logger.error(f"Error processing query: {str(e)}")
        return VHReasonsResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )
    

@agent.on_rest_post("/vh_agent", VHAgentRequest, VHAgentResponse)
async def handle_vh_agent(ctx: Context, msg: VHAgentRequest) -> VHAgentResponse:
    try:
        ctx.logger.info(f"Received vaccine hesitancy insights agent query: {msg.query}")
        # TODO: Implement the agent logic here

        return VHAgentResponse(
            response="Not implemented",
            status="success"
        )
    except Exception as e:
        ctx.logger.error(f"Error processing query: {str(e)}")
        return VHAgentResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

def sql_escape(value):
    if value is None:
        return None
    return str(value).replace("'", "''")


if __name__ == "__main__":
    agent.run()
