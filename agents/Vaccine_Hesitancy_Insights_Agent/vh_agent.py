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
import mcp
from mcp.client.streamable_http import streamablehttp_client
import json
import base64
import asyncio
from typing import Dict, Any, List
from contextlib import AsyncExitStack
import os

# Load environment variables
load_dotenv()

SUPABASE_ACCESS_TOKEN = os.getenv("SUPABASE_ACCESS_TOKEN")
SMITHERY_API_KEY = os.getenv("SMITHERY_API_KEY")
SUPABASE_PROJECT_ID = os.getenv("SUPABASE_PROJECT_ID")
ASI1_API_KEY = os.getenv("ASI1_API_KEY")

if not SUPABASE_ACCESS_TOKEN or not SMITHERY_API_KEY or not SUPABASE_PROJECT_ID or not ASI1_API_KEY:
    raise ValueError("Missing required environment variables.")

class SupabaseMCPClient:
    def __init__(self):
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.config = {
            "accessToken": SUPABASE_ACCESS_TOKEN,
            "readOnly": True,
        }
        self.project_id = SUPABASE_PROJECT_ID

    async def connect(self, ctx: Context):
        config_b64 = base64.b64encode(json.dumps(self.config).encode())
        url = f"https://server.smithery.ai/@supabase-community/supabase-mcp/mcp?config={config_b64}&api_key={SMITHERY_API_KEY}&profile=dual-barnacle-C2qHG5"
        read_stream, write_stream, _ = await self.exit_stack.enter_async_context(
            streamablehttp_client(url)
        )
        self.session = await self.exit_stack.enter_async_context(
            mcp.ClientSession(read_stream, write_stream)
        )
        await self.session.initialize()
        ctx.logger.info("Connected to Supabase MCP server")

    async def ensure_connection(self, ctx: Context):
        if not self.session:
            await self.connect(ctx)
            return
        
        try:
            await self.session.list_tools()
        except Exception as e:
            ctx.logger.warning(f"Session check failed: {str(e)}. Attempting to reconnect...")
            await self.cleanup()
            await self.connect(ctx)

    async def call_tool(self, tool_name: str, arguments: dict, ctx: Context):
        await self.ensure_connection(ctx)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return await self.session.call_tool(tool_name, arguments=arguments)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                ctx.logger.warning(f"Tool call attempt {attempt + 1} failed: {str(e)}. Retrying...")
                await asyncio.sleep(1)
                await self.ensure_connection(ctx)

    async def cleanup(self):
        await self.exit_stack.aclose()
        self.session = None

# ASI1 configuration
ASI1_URL = "https://api.asi1.ai/v1/chat/completions"
ASI1_HEADERS = {
    "Authorization": f"Bearer {ASI1_API_KEY}",
    "Content-Type": "application/json"
}

class VaccineHesitancyAgent:
    def __init__(self, mcp_client: SupabaseMCPClient):
        self.mcp_client = mcp_client
        
    async def query_ons_data(self, refined_query: str, ctx: Context) -> Dict[str, Any]:
        """ONS Tool: Specialized for vaccine hesitancy demographic data with ASI1 SQL generation"""
        
        ons_system_prompt = f"""You are an ONS vaccine hesitancy data specialist. Generate intelligent SQL queries for these tables:

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
vaccine_hesitancy_age_sex: wave_date, group, subgroup, measure, value, weighted_count, sample_size
vaccine_hesitancy_trends: period, block, measure, value_type, value, weighted_count, sample_size
vaccine_hesitancy_barriers: block, group, measure, value_type, value, weighted_count, sample_size
vaccine_hesitancy_reasons: period, group, block, measure, percent, lcl, ucl, weighted_count, sample_size

**INTELLIGENT SQL RULES:**
- Use simple table names: vaccine_hesitancy_sex NOT `project.vaccine_hesitancy_sex`
- NO backticks, NO project prefixes, NO schemas
- Always include LIMIT to avoid large results

**QUERY STRATEGY:**
1. **For GENERAL queries** (e.g., "vaccine hesitancy by gender"): Use exploratory SELECT * 
2. **For SPECIFIC queries** (e.g., "North West", "Male", "Muslim"): Add WHERE clause with the specific value

**EXAMPLES:**

**General Query:** "vaccine hesitancy by gender"
→ `SELECT * FROM vaccine_hesitancy_sex LIMIT 10;`

**Specific Query:** "vaccine hesitancy in North West"  
→ `SELECT * FROM vaccine_hesitancy_region WHERE region = 'North West' LIMIT 10;`

**Specific Query:** "vaccine hesitancy among males"
→ `SELECT * FROM vaccine_hesitancy_sex WHERE sex = 'Male' LIMIT 10;`

**Specific Query:** "vaccine hesitancy in London"
→ `SELECT * FROM vaccine_hesitancy_region WHERE region = 'London' LIMIT 10;`

**Multiple Specific Values:** "vaccine hesitancy in North West and Scotland"  
→ `SELECT * FROM vaccine_hesitancy_region WHERE region IN ('North West', 'Scotland') LIMIT 20;`

**AVAILABLE REGION VALUES:** "South East", "All adults", "Scotland", "Wales", "East of England", "North West", "North East", "England", "South West", "London"

Generate ONLY the SQL query, no explanation. Be intelligent about specificity!"""

        try:
            payload = {
                "model": "asi1-mini",
                "messages": [
                    {"role": "system", "content": ons_system_prompt},
                    {"role": "user", "content": f"Generate SQL query for: {refined_query}"}
                ],
                "temperature": 0.1,
                "max_tokens": 200
            }
            
            response = requests.post(ASI1_URL, headers=ASI1_HEADERS, json=payload)
            if response.status_code != 200:
                return {"error": f"ASI1 API error: {response.status_code}", "query": refined_query}
            
            sql_query = response.json()["choices"][0]["message"]["content"].strip()
            
            # Clean up the SQL query - remove any markdown formatting or extra text
            if "```sql" in sql_query:
                sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql_query:
                sql_query = sql_query.split("```")[1].strip()
            
            # Remove any backticks or project prefixes that might slip through
            sql_query = sql_query.replace("`", "").replace(f"{SUPABASE_PROJECT_ID}.", "")
            
            ctx.logger.info(f"ONS SQL generated: {sql_query}")
            
            # Execute SQL via MCP
            result = await self.mcp_client.call_tool("execute_sql", {
                "project_id": SUPABASE_PROJECT_ID,
                "query": sql_query
            }, ctx)
            
            content = result.content[0].text if isinstance(result.content, list) else result.content
            return {
                "success": True,
                "data": json.loads(content),
                "sql_query": sql_query,
                "source": "ONS"
            }
            
        except Exception as e:
            ctx.logger.error(f"ONS tool error: {str(e)}")
            return {"error": str(e), "query": refined_query, "source": "ONS"}

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
                return {"error": f"ASI1 API error: {response.status_code}", "query": refined_query}
            
            sql_query = response.json()["choices"][0]["message"]["content"].strip()
            
            # Clean up the SQL query - remove any markdown formatting or extra text
            if "```sql" in sql_query:
                sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql_query:
                sql_query = sql_query.split("```")[1].strip()
            
            # Remove any backticks or project prefixes that might slip through
            sql_query = sql_query.replace("`", "").replace(f"{SUPABASE_PROJECT_ID}.", "")
            
            ctx.logger.info(f"Twitter SQL generated: {sql_query}")
            
            # Execute SQL via MCP
            result = await self.mcp_client.call_tool("execute_sql", {
                "project_id": SUPABASE_PROJECT_ID,
                "query": sql_query
            }, ctx)
            
            content = result.content[0].text if isinstance(result.content, list) else result.content
            return {
                "success": True,
                "data": json.loads(content),
                "sql_query": sql_query,
                "source": "Twitter"
            }
            
        except Exception as e:
            ctx.logger.error(f"Twitter tool error: {str(e)}")
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

    async def format_response(self, routing_result: Dict, tool_results: List[Dict], ctx: Context) -> str:
        """Format raw SQL results into human-readable response"""
        
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

            # Format successful results
            response_parts = []
            
            for result in tool_results:
                if result.get("success") and result.get("data"):
                    source = result.get("source", "Unknown")
                    data = result["data"]
                    sql_query = result.get("sql_query", "")
                    
                    if data:
                        response_parts.append(f"**{source} Data:**")
                        
                        # Enhanced formatting based on data type and source
                        if isinstance(data, list) and data:
                            # Determine appropriate display limit based on data type
                            should_show_all = self._should_show_all_records(source, data)
                            display_limit = self._get_display_limit(source, len(data))
                            
                            if should_show_all or len(data) <= display_limit:
                                # Show all records
                                for i, row in enumerate(data, 1):
                                    formatted_row = self._format_row(row, source)
                                    response_parts.append(f"{i}. {formatted_row}")
                            else:  # Summarize if too many rows
                                sample_size = min(8, len(data))  # Show more samples
                                response_parts.append(f"Found {len(data)} records. Showing first {sample_size}:")
                                for i, row in enumerate(data[:sample_size], 1):
                                    formatted_row = self._format_row(row, source)
                                    response_parts.append(f"{i}. {formatted_row}")
                                response_parts.append(f"... and {len(data) - sample_size} more records")
                        else:
                            response_parts.append("No data found for your query.")
                        
                        # Show refined query for transparency
                        response_parts.append(f"*Query executed: {sql_query}*")
                        response_parts.append("")
            
            if not response_parts:
                return "No data was found for your query. Please try a different search or check the available data sources."
            
            return "\n".join(response_parts)
            
        except Exception as e:
            ctx.logger.error(f"Response formatting error: {str(e)}")
            return f"Error formatting response: {str(e)}"

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
    mailbox=True
)

client = SupabaseMCPClient()
vh_agent = VaccineHesitancyAgent(client)

@agent.on_event("startup")
async def startup_function(ctx: Context):
    ctx.logger.info("🚀 Starting Vaccine Hesitancy Insights Agent")
    ctx.logger.info("📊 Capabilities: ONS demographic data + Twitter sentiment analysis")
    try:
        await client.connect(ctx)
        ctx.logger.info("✅ Connected to Supabase MCP server")
    except Exception as e:
        ctx.logger.error(f"❌ Failed to connect to MCP server: {str(e)}")
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
        
        # Send acknowledgement
        ack = ChatAcknowledgement(
            timestamp=datetime.now(timezone.utc),
            acknowledged_msg_id=msg.msg_id
        )
        await ctx.send(sender, ack)
        
        # Extract user message
        user_message = next((item.text for item in msg.content if isinstance(item, TextContent)), None)
        if not user_message:
            ctx.logger.warning("No text content found in message")
            return
        
        ctx.logger.info(f"🔍 Processing query: {user_message}")
        
        # Process query through the main agent
        response_text = await vh_agent.process_query(user_message, ctx)
        
        # Send response
        response_msg = ChatMessage(
            timestamp=datetime.now(timezone.utc),
            msg_id=uuid4(),
            content=[TextContent(type="text", text=response_text)]
        )
        
        await ctx.send(sender, response_msg)
        ctx.logger.info(f"✅ Sent response message {response_msg.msg_id}")
        
    except Exception as e:
        ctx.logger.error(f"❌ Error handling chat message: {str(e)}")
        error_response = ChatMessage(
            timestamp=datetime.now(timezone.utc),
            msg_id=uuid4(),
            content=[TextContent(type="text", text=f"An error occurred: {str(e)}")]
        )
        await ctx.send(sender, error_response)

@chat_proto.on_message(model=ChatAcknowledgement)
async def handle_chat_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"📝 Received acknowledgement from {sender} for message {msg.acknowledged_msg_id}")

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
