import requests
from dotenv import load_dotenv
from uagents_core.contrib.protocols.chat import (
    chat_protocol_spec,
    ChatMessage,
    ChatAcknowledgement,
    TextContent,
    StartSessionContent,
)
from uagents import Agent, Context, Protocol
from datetime import datetime, timezone
from uuid import uuid4
import mcp
from mcp.client.streamable_http import streamablehttp_client
import json
import base64
import asyncio
from typing import Dict, Any
from contextlib import AsyncExitStack
import os
from hesitancy_models import VHAgentRequest, VHAgentResponse

# Load environment variables
load_dotenv()

SUPABASE_ACCESS_TOKEN = os.getenv("SUPABASE_ACCESS_TOKEN")
SMITHERY_API_KEY = os.getenv("SMITHERY_API_KEY")
SUPABASE_PROJECT_ID = os.getenv("SUPABASE_PROJECT_ID")
ASI1_API_KEY = os.getenv("ASI1_API_KEY")

if not SUPABASE_ACCESS_TOKEN or not SMITHERY_API_KEY or not SUPABASE_PROJECT_ID or not ASI1_API_KEY:
    raise ValueError("Missing required environment variables.")

print(f"Loaded project ID from env: {SUPABASE_PROJECT_ID}")  # Debug print

class SupabaseMCPClient:
    def __init__(self):
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.config = {
            "accessToken": SUPABASE_ACCESS_TOKEN,
            "readOnly": True,
        }
        self.project_id = SUPABASE_PROJECT_ID
        print(f"Initialized MCP client with project ID: {self.project_id}")  # Debug print

    async def connect(self, ctx: Context):
        config_b64 = base64.b64encode(json.dumps(self.config).encode())
        url = f"https://server.smithery.ai/@supabase-community/supabase-mcp/mcp?config={config_b64}&api_key={SMITHERY_API_KEY}&profile=dual-barnacle-C2qHG5"
        ctx.logger.info(f"Connecting to MCP with project ID: {self.project_id}")  # Debug log
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

    async def call_tool(self, tool_name: str, arguments: dict, ctx: Context):
        # Ensure we have a fresh connection before each tool call
        await self.ensure_connection(ctx)
        ctx.logger.info(f"Calling tool {tool_name} with arguments: {arguments}")  # Debug log
        
        # Add retry logic for transient failures
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return await self.session.call_tool(tool_name, arguments=arguments)
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise  # Re-raise the last exception
                ctx.logger.warning(f"Tool call attempt {attempt + 1} failed: {str(e)}. Retrying...")
                await asyncio.sleep(1)  # Wait 1 second before retry
                await self.ensure_connection(ctx)  # Refresh connection for retry

    async def cleanup(self):
        await self.exit_stack.aclose()
        self.session = None

# Set up chat protocol and agent
chat_proto = Protocol(spec=chat_protocol_spec)
agent = Agent(
    name='hesitancy_insights_agent',
    seed="hesitancy_insights_agent",
    port=8001,
    mailbox=True
)
client = SupabaseMCPClient()

@agent.on_event("startup")
async def startup_function(ctx: Context):
    ctx.logger.info("Starting up Hesitancy Insights Chat Agent")
    try:
        await client.connect(ctx)
        ctx.logger.info("Successfully connected to MCP server")
    except Exception as e:
        ctx.logger.error(f"Failed to connect to MCP server: {str(e)}")
        raise

ASI1_URL = "https://api.asi1.ai/v1/chat/completions"
ASI1_HEADERS = {
    "Authorization": f"Bearer {ASI1_API_KEY}",
    "Content-Type": "application/json"
}

# Tool schemas for ASI1
execute_sql_tool = {
    "type": "function",
    "function": {
        "name": "execute_sql",
        "description": "Run a SQL query on the vaccine hesitancy Supabase database.",
        "parameters": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "query": {"type": "string"}
            },
            "required": ["project_id", "query"]
        }
    }
}
list_tables_tool = {
    "type": "function",
    "function": {
        "name": "list_tables",
        "description": "List all tables in the Supabase database. Use this to refresh the table list.",
        "parameters": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string"}
            },
            "required": ["project_id"]
        }
    }
}

SYSTEM_PROMPT = {
    "role": "system",
    "content": f"""You are a vaccine hesitancy data assistant with access to a Supabase database. You have two tools available:

1. list_tables: Use this tool ONLY when you need to refresh the list of available tables in the database. Do NOT use this tool for querying data.

2. execute_sql: Use this tool when you need to:
   - Query data from specific tables
   - Get rows from tables
   - Run any SQL queries
   - When the user asks for data from a specific table

AVAILABLE TABLES AND THEIR COLUMNS:

vaccine_hesitancy_region: region (text), measure (text), percent (double precision), weighted_count (double precision), sample_size (double precision)
vaccine_hesitancy_age: wave_date (text), age_band (text), measure (text), percent (double precision), lcl (double precision), ucl (double precision), weighted_count (double precision), sample_size (double precision)
vaccine_hesitancy_age_group: wave_date (text), age_group (text), measure (text), percent (double precision), lcl (double precision), ucl (double precision), weighted_count (double precision), sample_size (double precision)
vaccine_hesitancy_sex: wave_date (text), sex (text), measure (text), percent (double precision), lcl (double precision), ucl (double precision), weighted_count (double precision), sample_size (double precision)
vaccine_hesitancy_ethnicity: wave_date (text), ethnicity (text), measure (text), percent (double precision), lcl (double precision), ucl (double precision), weighted_count (double precision), sample_size (double precision)
vaccine_hesitancy_religion: wave_date (text), religion (text), measure (text), percent (double precision), lcl (double precision), ucl (double precision), weighted_count (double precision), sample_size (double precision)
vaccine_hesitancy_disability: wave_date (text), disability_status (text), measure (text), percent (double precision), lcl (double precision), ucl (double precision), weighted_count (double precision), sample_size (double precision)
vaccine_hesitancy_cev: wave_date (text), cev_status (text), measure (text), percent (double precision), lcl (double precision), ucl (double precision), weighted_count (double precision), sample_size (double precision)
vaccine_hesitancy_health_condition: wave_date (text), health_condition (text), measure (text), percent (double precision), lcl (double precision), ucl (double precision), weighted_count (double precision), sample_size (double precision)
vaccine_hesitancy_health_general_condition: wave_date (text), health_general_condition (text), measure (text), percent (double precision), lcl (double precision), ucl (double precision), weighted_count (double precision), sample_size (double precision)
vaccine_hesitancy_imd_quintile: wave_date (text), imd_quintile (text), measure (text), percent (double precision), lcl (double precision), ucl (double precision), weighted_count (double precision), sample_size (double precision)
vaccine_hesitancy_employment: wave_date (text), employment_status (text), measure (text), percent (double precision), lcl (double precision), ucl (double precision), weighted_count (double precision), sample_size (double precision)
vaccine_hesitancy_expense_affordability: wave_date (text), expense_affordability (text), measure (text), percent (double precision), lcl (double precision), ucl (double precision), weighted_count (double precision), sample_size (double precision)
vaccine_hesitancy_household_type: wave_date (text), household_type (text), measure (text), percent (double precision), lcl (double precision), ucl (double precision), weighted_count (double precision), sample_size (double precision)
vaccine_hesitancy_caregiver_status: wave_date (text), caregiver_status (text), measure (text), percent (double precision), lcl (double precision), ucl (double precision), weighted_count (double precision), sample_size (double precision)
vaccine_hesitancy_age_sex: wave_date (text), group (text), subgroup (text), measure (text), value (double precision), weighted_count (double precision), sample_size (double precision)
vaccine_hesitancy_trends: period (text), block (text), measure (text), value_type (text), value (double precision), weighted_count (double precision), sample_size (double precision)
vaccine_hesitancy_barriers: block (text), group (text), measure (text), value_type (text), value (double precision), weighted_count (double precision), sample_size (double precision)
vaccine_hesitancy_reasons: period (text), group (text), block (text), measure (text), percent (double precision), lcl (double precision), ucl (double precision), weighted_count (double precision), sample_size (double precision)
vaccine_tweets: tweet_id (character varying), content (text), author_username (character varying), author_name (character varying), author_profile_image (text), created_at (timestamp with time zone), fetched_at (timestamp with time zone), likes (integer), retweets (integer), replies (integer), sentiment (character varying), impact_score (double precision), engagement_score (double precision)

EXAMPLE DATA VALUES:
- Available regions: "South East", "All adults", "Scotland", "Wales", "East of England", "North West", "North East", "England", "South West", "London"
- Available measures: "Have received a vaccine (one or two doses)", "Have been offered a vaccine and waiting to be vaccinated", "Have been offered a vaccine but declined the offer", etc.

IMPORTANT: 
- Always use the project ID '{SUPABASE_PROJECT_ID}' for all tool calls
- You have access to the list of available tables above - use these exact table names in your queries
- Use ONLY the columns listed above for each table - do NOT use columns that don't exist
- Most tables use 'wave_date' for time data, not 'date'
- vaccine_hesitancy_region has NO date column - only region, measure, percent, weighted_count, sample_size
- If the user asks for data from a specific table (like 'show me rows from X' or 'get data from X'), use execute_sql with the correct table name and columns
- Only use list_tables when you need to refresh the table list
- For any data query, use execute_sql with appropriate SQL query using the exact table names and columns from the list above
- You have access to the user's chat history, so you can reference previous questions and provide contextual responses
- If a user asks follow-up questions like "what about that data?" or "can you show me more details?", use the context from previous messages to understand what they're referring to
- If a user asks about a table that's not in the list above, suggest using list_tables to refresh the table list first"""
}

@chat_proto.on_message(model=ChatMessage)
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    try:
        ctx.logger.info(f"Received message from {sender}")
        
        # Check if we've already processed this message
        processed_key = f"processed_{msg.msg_id}"
        if ctx.storage.get(processed_key):
            ctx.logger.info(f"Message {msg.msg_id} already processed, skipping")
            return
        
        # Mark this message as processed
        ctx.storage.set(processed_key, True)
        
        ack = ChatAcknowledgement(
            timestamp=datetime.now(timezone.utc),
            acknowledged_msg_id=msg.msg_id
        )
        await ctx.send(sender, ack)
        ctx.logger.info(f"Sent acknowledgement for message {msg.msg_id}")

        if not client.session:
            ctx.logger.info("No active MCP session, attempting to connect...")
            await client.connect(ctx)

        # Get user message
        user_message = next((item.text for item in msg.content if isinstance(item, TextContent)), None)
        if not user_message:
            ctx.logger.warning("No text content found in message")
            return
        ctx.logger.info(f"Processing user message: {user_message}")

        # Get existing chat history for this sender
        chat_history_key = f"chat_history_{sender}"
        chat_history = ctx.storage.get(chat_history_key) or []
        
        # Add current user message to history
        chat_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Keep only last 10 messages to prevent memory issues
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]
        
        # Store updated history
        ctx.storage.set(chat_history_key, chat_history)

        # Build messages with chat history
        messages = [SYSTEM_PROMPT]
        
        # Add chat history (excluding current message)
        for msg in chat_history[:-1]:  # Exclude the current message
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current message
        messages.append({"role": "user", "content": user_message})

        # 1. First call to ASI1 with tools and history
        payload = {
            "model": "asi1-mini",
            "messages": messages,
            "tools": [execute_sql_tool, list_tables_tool],
            "temperature": 0.2,
            "max_tokens": 1024
        }
        
        ctx.logger.info("Making first call to ASI1")
        response = requests.post(ASI1_URL, headers=ASI1_HEADERS, json=payload)
        resp_json = response.json()
        ctx.logger.info("Received response from ASI1")

        # 2. Check for tool call
        tool_calls = resp_json["choices"][0]["message"].get("tool_calls", [])
        if tool_calls:
            tool_call = tool_calls[0]  # Only handle the first tool call for now
            function_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            ctx.logger.info(f"Tool call requested: {function_name} with arguments: {arguments}")
            
            try:
                # Always use our project ID, ignore what ASI1 provides
                arguments["project_id"] = SUPABASE_PROJECT_ID
                ctx.logger.info(f"Using project_id: {SUPABASE_PROJECT_ID}")

                result = await client.call_tool(function_name, arguments, ctx)
                tool_result = result.content[0].text if isinstance(result.content, list) else result.content
                ctx.logger.info(f"Tool call successful, result: {tool_result[:100]}...")  # Log first 100 chars
            except Exception as e:
                ctx.logger.error(f"Error calling tool: {str(e)}")
                tool_result = f"Error calling tool: {str(e)}"
            
            # Add tool result message
            tool_result_message = {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_result
            }
            
            # Final call to ASI1 for the answer
            ctx.logger.info("Making final call to ASI1 with tool result")
            final_payload = {
                "model": "asi1-mini",
                "messages": [
                    SYSTEM_PROMPT,
                    {"role": "user", "content": user_message},
                    resp_json["choices"][0]["message"],
                    tool_result_message
                ],
                "temperature": 0.2,
                "max_tokens": 1024
            }
            
            final_response = requests.post(ASI1_URL, headers=ASI1_HEADERS, json=final_payload)
            final_json = final_response.json()
            response_text = final_json["choices"][0]["message"]["content"]
            ctx.logger.info("Received final response from ASI1")
        else:
            response_text = resp_json["choices"][0]["message"]["content"]
            ctx.logger.info("No tool calls requested, using direct response")

        # Add assistant response to chat history
        chat_history.append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Keep only last 10 messages (including the new assistant response)
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]
        
        # Store updated history
        ctx.storage.set(chat_history_key, chat_history)

        response_msg = ChatMessage(
            timestamp=datetime.now(timezone.utc),
            msg_id=uuid4(),
            content=[TextContent(type="text", text=response_text)]
        )
        await ctx.send(sender, response_msg)
        ctx.logger.info(f"Sent response message {response_msg.msg_id}")
    except Exception as e:
        ctx.logger.error(f"Error handling chat message: {str(e)}")
        error_response = ChatMessage(
            timestamp=datetime.now(timezone.utc),
            msg_id=uuid4(),
            content=[TextContent(type="text", text=f"An error occurred: {str(e)}")]
        )
        await ctx.send(sender, error_response)
        ctx.logger.info("Sent error response message")

@chat_proto.on_message(model=ChatAcknowledgement)
async def handle_chat_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"Received acknowledgement from {sender} for message {msg.acknowledged_msg_id}")
    if msg.metadata:
        ctx.logger.info(f"Metadata: {msg.metadata}")

agent.include(chat_proto)

if __name__ == "__main__":
    try:
        agent.run()
    except Exception as e:
        print(f"Error running agent: {str(e)}")
    finally:
        asyncio.run(client.cleanup())




