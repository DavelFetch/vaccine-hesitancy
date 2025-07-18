from dotenv import load_dotenv
from uagents import Agent, Context, Model
from uagents.setup import fund_agent_if_low
from datetime import datetime, timezone, timedelta
import mcp
from mcp.client.streamable_http import streamablehttp_client
import json
import base64
import asyncio
from typing import Dict, List, Optional, Any
from contextlib import AsyncExitStack
import os
import re
import requests
from x_analysis_models import (
    ScrapeTweetsRequest, ScrapeTweetsResponse,
    ScrapeMultipleQueriesRequest, ScrapeMultipleQueriesResponse,
    SentimentRequest, SentimentResponse,
    TrendRequest, TrendResponse,
    TweetsRequest, TweetsResponse,
    UserAnalysisRequest, UserAnalysisResponse,
    TrendingTopicsRequest, TrendingTopicsResponse,
    DashboardDataRequest, DashboardDataResponse,
    SentimentTimelineRequest, SentimentTimelineResponse,
    TopUsersRequest, TopUsersResponse,
    TweetData
)
# OAuth import removed - using RapidAPI now

# Add debug response model
class DebugResponse(Model):
    response: str
    status: str
    data: str = ""

# Load environment variables
load_dotenv()

# Validate required environment variables - Updated for RapidAPI
required_env_vars = {
    "SUPABASE_ACCESS_TOKEN": os.getenv("SUPABASE_ACCESS_TOKEN"),
    "SMITHERY_API_KEY": os.getenv("SMITHERY_API_KEY"),
    "SUPABASE_PROJECT_ID": os.getenv("SUPABASE_PROJECT_ID"),
    "ASI1_API_KEY": os.getenv("ASI1_API_KEY"),
    "RAPIDAPI_KEY": os.getenv("RAPIDAPI_KEY")
}

# Check if any required variables are missing
missing_vars = [var for var, value in required_env_vars.items() if not value]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# ASI1 configuration
ASI1_URL = "https://api.asi1.ai/v1/chat/completions"
ASI1_HEADERS = {
    "Authorization": f"Bearer {required_env_vars['ASI1_API_KEY']}",
    "Content-Type": "application/json"
}

# RapidAPI configuration
RAPIDAPI_KEY = required_env_vars["RAPIDAPI_KEY"]
RAPIDAPI_HOST = "twitter-v24.p.rapidapi.com"
RAPIDAPI_BASE_URL = f"https://{RAPIDAPI_HOST}"
RAPIDAPI_HEADERS = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": RAPIDAPI_HOST
}

def rapidapi_search_tweets(query, max_results=10):
    """
    Search recent tweets using RapidAPI "The Old Bird V2" API
    Returns tweets in the new format with globalObjects structure
    """
    url = f"{RAPIDAPI_BASE_URL}/search/"
    params = {
        "query": query,
        "section": "top",  # Can be "top", "latest", "people", "photos", "videos"
        "limit": max(10, min(max_results, 100))
    }
    
    try:
        resp = requests.get(url, headers=RAPIDAPI_HEADERS, params=params)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"RapidAPI error: {str(e)}")
        raise

def parse_rapidapi_response(rapidapi_data):
    """
    Convert RapidAPI response format to a format compatible with existing code
    Returns: dict with 'data' (tweets array) and 'includes' (users data)
    """
    try:
        # Check if response has globalObjects structure (expected format)
        if 'globalObjects' in rapidapi_data:
            print(f"DEBUG: Found globalObjects structure")
            global_objects = rapidapi_data.get('globalObjects', {})
            tweets_obj = global_objects.get('tweets', {})
            users_obj = global_objects.get('users', {})
            
            # Convert tweets from object to array format
            tweets_array = []
            for tweet_id, tweet_data in tweets_obj.items():
                # Map RapidAPI fields to Twitter API v2 format
                converted_tweet = {
                    'id': tweet_data.get('id_str', tweet_id),
                    'text': tweet_data.get('full_text', ''),
                    'author_id': tweet_data.get('user_id_str', ''),
                    'created_at': convert_twitter_date(tweet_data.get('created_at', '')),
                    'public_metrics': {
                        'like_count': tweet_data.get('favorite_count', 0),
                        'retweet_count': tweet_data.get('retweet_count', 0),
                        'reply_count': tweet_data.get('reply_count', 0),
                        'quote_count': tweet_data.get('quote_count', 0),
                        'bookmark_count': tweet_data.get('bookmark_count', 0),
                        'impression_count': extract_view_count(tweet_data)
                    }
                }
                tweets_array.append(converted_tweet)
            
            # Convert users from object to array format
            users_array = []
            for user_id, user_data in users_obj.items():
                converted_user = {
                    'id': user_data.get('id_str', user_id),
                    'username': user_data.get('screen_name', ''),
                    'name': user_data.get('name', ''),
                    'profile_image_url': user_data.get('profile_image_url_https', '')
                }
                users_array.append(converted_user)
            
            return {
                'data': tweets_array,
                'includes': {
                    'users': users_array
                }
            }
        
        # Check if response has 'data' key with timeline structure (RapidAPI format)
        elif 'data' in rapidapi_data:
            print(f"DEBUG: Found 'data' key structure")
            data_obj = rapidapi_data.get('data', {})
            print(f"DEBUG: Data object type: {type(data_obj)}")
            
            # Navigate the nested timeline structure
            tweets_array = []
            users_set = set()
            users_array = []
            
            try:
                # Expected path: data.search_by_raw_query.search_timeline.timeline.instructions
                search_data = data_obj.get('search_by_raw_query', {})
                search_timeline = search_data.get('search_timeline', {})
                timeline = search_timeline.get('timeline', {})
                instructions = timeline.get('instructions', [])
                
                print(f"DEBUG: Found {len(instructions)} timeline instructions")
                
                for instruction in instructions:
                    if instruction.get('type') == 'TimelineAddEntries':
                        entries = instruction.get('entries', [])
                        print(f"DEBUG: Processing {len(entries)} timeline entries")
                        
                        for entry in entries:
                            content = entry.get('content', {})
                            if content.get('entryType') == 'TimelineTimelineItem':
                                item_content = content.get('itemContent', {})
                                if item_content.get('itemType') == 'TimelineTweet':
                                    tweet_results = item_content.get('tweet_results', {})
                                    tweet_result = tweet_results.get('result', {})
                                    
                                    if tweet_result.get('__typename') == 'Tweet':
                                        # Extract tweet data
                                        tweet_id = tweet_result.get('rest_id', '')
                                        legacy = tweet_result.get('legacy', {})
                                        
                                        # Extract user data
                                        core = tweet_result.get('core', {})
                                        user_results = core.get('user_results', {})
                                        user_result = user_results.get('result', {})
                                        user_legacy = user_result.get('legacy', {})
                                        user_id = user_result.get('rest_id', '')
                                        
                                        # Build tweet object
                                        converted_tweet = {
                                            'id': tweet_id,
                                            'text': legacy.get('full_text', ''),
                                            'author_id': user_id,
                                            'created_at': convert_twitter_date(legacy.get('created_at', '')),
                                            'public_metrics': {
                                                'like_count': legacy.get('favorite_count', 0),
                                                'retweet_count': legacy.get('retweet_count', 0),
                                                'reply_count': legacy.get('reply_count', 0),
                                                'quote_count': legacy.get('quote_count', 0),
                                                'bookmark_count': legacy.get('bookmark_count', 0),
                                                'impression_count': 0  # Not available in this format
                                            }
                                        }
                                        tweets_array.append(converted_tweet)
                                        print(f"DEBUG: Extracted tweet {tweet_id}: {legacy.get('full_text', '')[:50]}...")
                                        
                                        # Extract user info if not already processed
                                        if user_id and user_id not in users_set:
                                            users_set.add(user_id)
                                            converted_user = {
                                                'id': user_id,
                                                'username': user_legacy.get('screen_name', ''),
                                                'name': user_legacy.get('name', ''),
                                                'profile_image_url': user_legacy.get('profile_image_url_https', '')
                                            }
                                            users_array.append(converted_user)
                                            print(f"DEBUG: Extracted user @{user_legacy.get('screen_name', 'unknown')}")
                
                print(f"DEBUG: Successfully converted {len(tweets_array)} tweets and {len(users_array)} users")
                return {
                    'data': tweets_array,
                    'includes': {
                        'users': users_array
                    }
                }
                
            except Exception as parse_error:
                print(f"DEBUG: Error parsing timeline structure: {str(parse_error)}")
                print(f"DEBUG: Timeline navigation failed, raw data sample: {str(data_obj)[:200]}...")
                
        # Check for any other potential structure
        else:
            print(f"DEBUG: Unknown structure. Available keys: {list(rapidapi_data.keys())}")
            print(f"DEBUG: Sample content: {str(rapidapi_data)[:300]}...")
        
        # Return empty if no recognized structure
        return {'data': [], 'includes': {'users': []}}
        
    except Exception as e:
        print(f"Error parsing RapidAPI response: {str(e)}")
        return {'data': [], 'includes': {'users': []}}

def convert_twitter_date(twitter_date_str):
    """
    Convert Twitter date format to ISO format
    From: "Fri May 19 21:55:37 +0000 2023"
    To: "2023-05-19T21:55:37.000Z"
    """
    try:
        from datetime import datetime
        # Parse Twitter's date format
        dt = datetime.strptime(twitter_date_str, "%a %b %d %H:%M:%S %z %Y")
        # Convert to ISO format
        return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    except Exception:
        return datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000Z")

def extract_view_count(tweet_data):
    """Extract view count from RapidAPI tweet data"""
    try:
        ext_views = tweet_data.get('ext_views', {})
        if ext_views.get('state') == 'EnabledWithCount':
            return int(ext_views.get('count', 0))
    except Exception:
        pass
    return 0

class XAnalysisMCPClient:
    def __init__(self):
        self.sessions: Dict[str, mcp.ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.config = {
            "accessToken": required_env_vars["SUPABASE_ACCESS_TOKEN"],
            "readOnly": False,  # We need write access for storing tweets
        }
        self.project_id = required_env_vars["SUPABASE_PROJECT_ID"]
        self.max_retries = 3
        self.retry_delay = 1  # seconds

    async def connect_supabase(self, ctx: Context):
        """Connect to Supabase MCP server"""
        try:
            config_b64 = base64.b64encode(json.dumps(self.config).encode()).decode()
            url = f"https://server.smithery.ai/@supabase-community/supabase-mcp/mcp?config={config_b64}&api_key={required_env_vars['SMITHERY_API_KEY']}&profile=dual-barnacle-C2qHG5"

            read_stream, write_stream, _ = await self.exit_stack.enter_async_context(
                streamablehttp_client(url)
            )
            session = await self.exit_stack.enter_async_context(
                mcp.ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            self.sessions["supabase"] = session
            ctx.logger.info("Connected to Supabase MCP server")
        except Exception as e:
            ctx.logger.error(f"Failed to connect to Supabase MCP: {str(e)}")
            raise

    async def ensure_connection(self, ctx: Context):
        """Ensure we have an active Supabase connection, reconnect if needed"""
        if "supabase" not in self.sessions:
            await self.connect_supabase(ctx)
            return

        try:
            # Try a simple operation to check if session is alive
            await self.sessions["supabase"].list_tools()
        except Exception as e:
            ctx.logger.warning(f"Supabase session check failed: {str(e)}. Attempting to reconnect...")
            await self.cleanup()  # Clean up all sessions
            await self.connect_supabase(ctx)  # Create new session

    async def cleanup(self):
        """Cleanup all resources"""
        await self.exit_stack.aclose()
        self.sessions.clear()

    async def ensure_table_exists(self, ctx: Context):
        """Ensure the vaccine_tweets table exists with correct schema"""
        try:
            create_table_query = """
                CREATE TABLE IF NOT EXISTS vaccine_tweets (
                    tweet_id VARCHAR(255) PRIMARY KEY,
                    content TEXT NOT NULL,
                    author_username VARCHAR(255),
                    author_name VARCHAR(255),
                    author_profile_image TEXT,
                    created_at TIMESTAMP WITH TIME ZONE,
                    fetched_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    likes INTEGER DEFAULT 0,
                    retweets INTEGER DEFAULT 0,
                    replies INTEGER DEFAULT 0,
                    sentiment VARCHAR(50) DEFAULT 'neutral',
                    impact_score FLOAT DEFAULT 0.0,
                    engagement_score FLOAT DEFAULT 0.0
                );
            """
            
            await self.execute_sql(create_table_query, ctx)
            ctx.logger.info("Ensured vaccine_tweets table exists")
            
            # Create indexes if they don't exist
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_vaccine_tweets_fetched_at ON vaccine_tweets(fetched_at);",
                "CREATE INDEX IF NOT EXISTS idx_vaccine_tweets_sentiment ON vaccine_tweets(sentiment);",
                "CREATE INDEX IF NOT EXISTS idx_vaccine_tweets_author_username ON vaccine_tweets(author_username);",
                "CREATE INDEX IF NOT EXISTS idx_vaccine_tweets_impact_score ON vaccine_tweets(impact_score);"
            ]
            
            for index_query in indexes:
                await self.execute_sql(index_query, ctx)
            
            ctx.logger.info("Ensured all indexes exist")
            
            # Test the table with a simple query
            test_query = "SELECT COUNT(*) as count FROM vaccine_tweets;"
            test_result = await self.execute_sql(test_query, ctx)
            if test_result and test_result.content:
                content = test_result.content[0].text if isinstance(test_result.content, list) else test_result.content
                ctx.logger.info(f"Table test query result: {content}")
            
        except Exception as e:
            ctx.logger.error(f"Error ensuring table exists: {str(e)}")
            raise

    async def execute_sql(self, query: str, ctx: Context):
        """Execute SQL query with retry logic"""
        for attempt in range(self.max_retries):
            try:
                await self.ensure_connection(ctx)
                return await self.sessions["supabase"].call_tool("execute_sql", arguments={
                    "project_id": self.project_id,
                    "query": query
                })
            except Exception as e:
                if attempt == self.max_retries - 1:  # Last attempt
                    raise  # Re-raise the last exception
                ctx.logger.warning(f"Query attempt {attempt + 1} failed: {str(e)}. Retrying...")
                await asyncio.sleep(self.retry_delay)

    def calculate_impact_score(self, likes: int, retweets: int, replies: int, quotes: int = 0) -> float:
        """Calculate impact score based on engagement metrics"""
        # Enhanced weighted formula: likes * 1 + retweets * 2 + replies * 3 + quotes * 2.5
        return (likes * 1) + (retweets * 2) + (replies * 3) + (quotes * 2.5)

    def calculate_engagement_score(self, likes: int, retweets: int, replies: int, quotes: int = 0, bookmarks: int = 0) -> float:
        """Calculate engagement score (percentage) with enhanced metrics"""
        total_engagement = likes + retweets + replies + quotes + bookmarks
        # Normalize to a 0-100 scale (assuming max engagement of 10000)
        return min((total_engagement / 10000) * 100, 100)

    async def store_tweets_batch(self, tweets_data: List[Dict], ctx: Context, includes_data: Dict = None) -> int:
        """Store multiple tweets with batch sentiment analysis and enhanced author information"""
        try:
            if not tweets_data:
                return 0
            
            # Create a mapping of user IDs to user data from includes
            user_map = {}
            if includes_data and 'users' in includes_data:
                for user in includes_data['users']:
                    user_map[user['id']] = user
                    ctx.logger.info(f"User mapping: {user['id']} -> {user.get('username', 'unknown')}")
            
            # Extract tweet contents for batch sentiment analysis
            tweet_contents = []
            for tweet in tweets_data:
                content = tweet.get('text', '')
                tweet_contents.append(content)
            
            # Analyze sentiment for all tweets in batch
            sentiments = await self.analyze_sentiment_batch(tweet_contents, ctx)
            
            # Store tweets with their analyzed sentiments
            stored_count = 0
            for i, tweet_data in enumerate(tweets_data):
                try:
                    # Parse tweet data - handle the actual structure from Twitter API
                    tweet_id = tweet_data.get('id', '')
                    content = tweet_data.get('text', '')
                    author_id = tweet_data.get('author_id', '')
                    
                    # Extract author information from includes data or fallback to text parsing
                    author_username = 'unknown_user'
                    author_name = 'Unknown User'
                    author_profile_image = ''
                    
                    # First try to get author info from the includes data
                    if author_id and author_id in user_map:
                        user_data = user_map[author_id]
                        author_username = user_data.get('username', 'unknown_user')
                        author_name = user_data.get('name', author_username)
                        author_profile_image = user_data.get('profile_image_url', '')
                        ctx.logger.info(f"Found author in includes: {author_username} ({author_name})")
                    else:
                        # Fallback to text parsing
                        # Try to extract username from retweet pattern
                        rt_match = re.search(r'RT @(\w+):', content)
                        if rt_match:
                            author_username = rt_match.group(1)
                            author_name = author_username  # Use username as name if no other info
                        else:
                            # Try to extract username from @username pattern
                            at_match = re.search(r'@(\w+)', content)
                            if at_match:
                                author_username = at_match.group(1)
                                author_name = author_username
                        
                        ctx.logger.info(f"Using fallback author parsing: {author_username}")
                    
                    # Parse timestamps with proper timezone handling
                    created_at = tweet_data.get('created_at', datetime.now(timezone.utc).isoformat())
                    fetched_at = datetime.now(timezone.utc).isoformat()
                    
                    # Extract ALL metrics from public_metrics
                    public_metrics = tweet_data.get('public_metrics', {})
                    ctx.logger.info(f"Enhanced public_metrics for tweet {tweet_id}: {public_metrics}")
                    
                    # Extract all available metrics
                    likes = public_metrics.get('like_count', 0)
                    retweets = public_metrics.get('retweet_count', 0)
                    replies = public_metrics.get('reply_count', 0)
                    quotes = public_metrics.get('quote_count', 0)  # Now available
                    bookmarks = public_metrics.get('bookmark_count', 0)  # Now available
                    impressions = public_metrics.get('impression_count', 0)  # Now available
                    
                    # Get sentiment from batch analysis
                    sentiment = sentiments[i] if i < len(sentiments) else "neutral"
                    impact_score = self.calculate_impact_score(likes, retweets, replies, quotes)
                    engagement_score = self.calculate_engagement_score(likes, retweets, replies, quotes, bookmarks)
                    
                    # Insert into database with enhanced metrics
                    query = f"""
                        INSERT INTO vaccine_tweets (
                            tweet_id, content, author_username, author_name, author_profile_image,
                            created_at, fetched_at, likes, retweets, replies,
                            sentiment, impact_score, engagement_score
                        ) VALUES (
                            '{tweet_id}', '{content.replace("'", "''")}', '{author_username}',
                            '{author_name.replace("'", "''")}', '{author_profile_image}',
                            '{created_at}', '{fetched_at}', {likes}, {retweets}, {replies},
                            '{sentiment}', {impact_score}, {engagement_score}
                        ) ON CONFLICT (tweet_id) DO NOTHING;
                    """
                    
                    try:
                        ctx.logger.info(f"Storing tweet {tweet_id}: likes={likes}, retweets={retweets}, replies={replies}, quotes={quotes}")
                        result = await self.execute_sql(query, ctx)
                        
                        # Check if the insert actually worked
                        if result and result.content:
                            content_result = result.content[0].text if isinstance(result.content, list) else result.content
                            ctx.logger.info(f"SQL result for tweet {tweet_id}: {content_result}")
                            
                            # Check if there was an error in the result
                            if "error" in content_result.lower():
                                ctx.logger.error(f"SQL error for tweet {tweet_id}: {content_result}")
                                continue  # Don't increment stored_count
                        
                        stored_count += 1
                        ctx.logger.info(f"Successfully stored tweet {tweet_id} (author: {author_username}, likes: {likes}, retweets: {retweets}, replies: {replies})")
                    except Exception as db_error:
                        ctx.logger.error(f"Database error storing tweet {tweet_id}: {str(db_error)}")
                        ctx.logger.error(f"Failed query: {query}")
                        # Continue with next tweet instead of failing completely
                        continue
                    
                except Exception as e:
                    ctx.logger.error(f"Error storing tweet {i}: {str(e)}")
                    continue
            
            return stored_count
            
        except Exception as e:
            ctx.logger.error(f"Error in batch tweet storage: {str(e)}")
            return 0

    async def verify_tweets_stored(self, tweet_ids: List[str], ctx: Context) -> int:
        """Verify that tweets were actually stored in the database"""
        try:
            if not tweet_ids:
                return 0
            
            # Create a query to check if tweets exist
            placeholders = ','.join([f"'{tweet_id}'" for tweet_id in tweet_ids])
            query = f"SELECT COUNT(*) as count FROM vaccine_tweets WHERE tweet_id IN ({placeholders});"
            
            result = await self.execute_sql(query, ctx)
            if result and result.content:
                content = result.content[0].text if isinstance(result.content, list) else result.content
                try:
                    data = json.loads(content)
                    count = data[0].get('count', 0) if data else 0
                    ctx.logger.info(f"Verification: {count} out of {len(tweet_ids)} tweets found in database")
                    return count
                except Exception as e:
                    ctx.logger.error(f"Error parsing verification result: {str(e)}")
                    return 0
            return 0
            
        except Exception as e:
            ctx.logger.error(f"Error verifying tweets: {str(e)}")
            return 0

    async def analyze_sentiment_batch(self, texts: List[str], ctx: Context) -> List[str]:
        """Analyze sentiment for multiple tweets in a single ASI1 API call"""
        try:
            if not texts:
                return []
            
            # Clean all texts
            cleaned_texts = []
            for text in texts:
                clean_text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
                clean_text = re.sub(r'@\w+|#\w+', '', clean_text)  # Remove mentions and hashtags
                clean_text = clean_text.strip()
                
                if clean_text and len(clean_text) >= 3:
                    cleaned_texts.append(clean_text)
                else:
                    cleaned_texts.append("")  # Keep empty for alignment
            
            if not any(cleaned_texts):
                return ["neutral"] * len(texts)
            
            # Create batch analysis prompt
            batch_prompt = {
                "role": "system",
                "content": """You are a sentiment analysis expert specializing in vaccine-related discussions. Analyze the sentiment of each text and respond with ONLY a comma-separated list of sentiments.

For each text, respond with one of these three words: 'positive', 'negative', or 'neutral'.

IMPORTANT CONTEXT RULES for vaccine discussions:
- Positive: Support for vaccines, positive health outcomes, trust in medical science, factual vaccine information
- Negative: Vaccine hesitancy, concerns about side effects, distrust in medical authorities, anti-vaccine rhetoric, conspiracy theories about vaccines, "Covid lies", "vaccine deception", government overreach in vaccine policy
- Neutral: Factual questions, balanced discussions, news reporting without clear stance

SPECIFIC PATTERNS TO WATCH FOR:
- "Covid lies" = negative (implies vaccine misinformation)
- "vaccine deception" = negative
- "government overreach" with vaccines = negative
- "banning states from labeling" = negative (implies government control)
- Enthusiasm about anti-vaccine content = negative (not positive)
- Questions about vaccine safety = usually negative (indicates concern)

Respond with only the sentiment words separated by commas, nothing else. Example: "positive,negative,neutral,positive" """
            }
            
            # Format texts for batch analysis
            numbered_texts = []
            for i, text in enumerate(cleaned_texts, 1):
                if text:
                    numbered_texts.append(f"Text {i}: {text}")
                else:
                    numbered_texts.append(f"Text {i}: [empty or too short]")
            
            user_message = {
                "role": "user", 
                "content": f"Analyze the sentiment of these texts:\n" + "\n".join(numbered_texts)
            }
            
            payload = {
                "model": "asi1-mini",
                "messages": [batch_prompt, user_message],
                "temperature": 0.1,  # Low temperature for consistent results
                "max_tokens": 50  # Increased for batch responses
            }
            
            ctx.logger.info(f"Analyzing sentiment for {len(texts)} tweets in batch...")
            response = requests.post(ASI1_URL, headers=ASI1_HEADERS, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                response_text = result["choices"][0]["message"]["content"].strip()
                
                # Parse comma-separated sentiments
                sentiments = [s.strip().lower() for s in response_text.split(',')]
                
                # Validate and fill results
                valid_sentiments = []
                for sentiment in sentiments:
                    if sentiment in ['positive', 'negative', 'neutral']:
                        valid_sentiments.append(sentiment)
                    else:
                        ctx.logger.warning(f"Invalid sentiment response: {sentiment}, defaulting to neutral")
                        valid_sentiments.append("neutral")
                
                # Ensure we have the right number of results
                while len(valid_sentiments) < len(texts):
                    valid_sentiments.append("neutral")
                
                # Truncate if we got too many
                valid_sentiments = valid_sentiments[:len(texts)]
                
                ctx.logger.info(f"Batch sentiment analysis completed: {len(valid_sentiments)} results")
                return valid_sentiments
            else:
                ctx.logger.error(f"ASI1 API error: {response.status_code} - {response.text}")
                return ["neutral"] * len(texts)
                
        except Exception as e:
            ctx.logger.error(f"Error in batch sentiment analysis: {str(e)}")
            return ["neutral"] * len(texts)

    async def analyze_sentiment_with_asi1(self, text: str, ctx: Context) -> str:
        """Analyze sentiment using ASI1 (single tweet)"""
        try:
            # Clean the text for analysis
            clean_text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
            clean_text = re.sub(r'@\w+|#\w+', '', clean_text)  # Remove mentions and hashtags
            clean_text = clean_text.strip()
            
            if not clean_text or len(clean_text) < 3:
                return "neutral"
            
            # Create sentiment analysis prompt
            sentiment_prompt = {
                "role": "system",
                "content": """You are a sentiment analysis expert specializing in vaccine-related discussions. Analyze the sentiment of the given text and respond with ONLY one of these three words: 'positive', 'negative', or 'neutral'.

IMPORTANT CONTEXT RULES for vaccine discussions:
- Positive: Support for vaccines, positive health outcomes, trust in medical science, factual vaccine information
- Negative: Vaccine hesitancy, concerns about side effects, distrust in medical authorities, anti-vaccine rhetoric, conspiracy theories about vaccines, "Covid lies", "vaccine deception", government overreach in vaccine policy
- Neutral: Factual questions, balanced discussions, news reporting without clear stance

SPECIFIC PATTERNS TO WATCH FOR:
- "Covid lies" = negative (implies vaccine misinformation)
- "vaccine deception" = negative
- "government overreach" with vaccines = negative
- "banning states from labeling" = negative (implies government control)
- Enthusiasm about anti-vaccine content = negative (not positive)
- Questions about vaccine safety = usually negative (indicates concern)

Respond with only the sentiment word, nothing else."""
            }
            
            user_message = {
                "role": "user", 
                "content": f"Analyze the sentiment of this text: '{clean_text}'"
            }
            
            payload = {
                "model": "asi1-mini",
                "messages": [sentiment_prompt, user_message],
                "temperature": 0.1,  # Low temperature for consistent results
                "max_tokens": 10
            }
            
            ctx.logger.info(f"Analyzing sentiment for text: {clean_text[:50]}...")
            response = requests.post(ASI1_URL, headers=ASI1_HEADERS, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                sentiment = result["choices"][0]["message"]["content"].strip().lower()
                
                # Validate sentiment response
                if sentiment in ['positive', 'negative', 'neutral']:
                    ctx.logger.info(f"Sentiment analysis result: {sentiment}")
                    return sentiment
                else:
                    ctx.logger.warning(f"Invalid sentiment response: {sentiment}, defaulting to neutral")
                    return "neutral"
            else:
                ctx.logger.error(f"ASI1 API error: {response.status_code} - {response.text}")
                return "neutral"
                
        except Exception as e:
            ctx.logger.error(f"Error in sentiment analysis: {str(e)}")
            return "neutral"

# Initialize agent and client
agent = Agent(
    name="x_analysis_rest_agent",
    port=8001,
    seed="x_analysis_rest_agent"
)
client = XAnalysisMCPClient()

@agent.on_event("startup")
async def startup_function(ctx: Context):
    ctx.logger.info("Starting up X Analysis Rest Agent")
    try:
        await client.connect_supabase(ctx)
        ctx.logger.info("✅ Supabase MCP connection established")
        await client.ensure_table_exists(ctx)
        ctx.logger.info("✅ Database table ensured")
    except Exception as e:
        ctx.logger.error(f"❌ Failed to initialize MCP connections: {str(e)}")
        raise

@agent.on_rest_post("/x/scrape", ScrapeTweetsRequest, ScrapeTweetsResponse)
async def handle_scrape_tweets(ctx: Context, msg: ScrapeTweetsRequest) -> ScrapeTweetsResponse:
    try:
        ctx.logger.info(f"Received tweet scraping request: query='{msg.query}'")
        max_results = 10
        try:
            rapidapi_response = rapidapi_search_tweets(msg.query, max_results)
            ctx.logger.info(f"RapidAPI response structure: {list(rapidapi_response.keys())}")
            
            # DEBUG: Log actual response content to see what we're getting
            ctx.logger.info(f"RapidAPI response content sample: {str(rapidapi_response)[:500]}...")
            
            # Parse RapidAPI response to compatible format
            tweets_data = parse_rapidapi_response(rapidapi_response)
            ctx.logger.info(f"Parsed response structure: {list(tweets_data.keys())}")
            
            # DEBUG: Log what we extracted
            if 'globalObjects' in rapidapi_response:
                global_objects = rapidapi_response['globalObjects']
                ctx.logger.info(f"GlobalObjects keys: {list(global_objects.keys())}")
                if 'tweets' in global_objects:
                    tweets_obj = global_objects['tweets']
                    ctx.logger.info(f"Raw tweets object keys count: {len(tweets_obj)}")
                    if tweets_obj:
                        first_tweet_id = list(tweets_obj.keys())[0]
                        ctx.logger.info(f"First tweet ID: {first_tweet_id}")
                        ctx.logger.info(f"First tweet sample: {str(tweets_obj[first_tweet_id])[:200]}...")
                else:
                    ctx.logger.warning("No 'tweets' key found in globalObjects")
            else:
                ctx.logger.warning("No 'globalObjects' key found in RapidAPI response")
                ctx.logger.info(f"Available keys: {list(rapidapi_response.keys())}")
        except Exception as e:
            ctx.logger.error(f"RapidAPI error: {str(e)}")
            return ScrapeTweetsResponse(
                response=f"RapidAPI error: {str(e)}",
                status="error",
                tweets_scraped=0
            )
        
        tweets = tweets_data.get('data', [])
        includes = tweets_data.get('includes', {})  # Get the includes data with user information
        
        ctx.logger.info(f"Extracted tweets count: {len(tweets) if tweets else 0}")
        ctx.logger.info(f"Includes data keys: {list(includes.keys()) if includes else 'None'}")
        
        if 'users' in includes:
            ctx.logger.info(f"Found {len(includes['users'])} users in includes data")
        
        if not tweets:
            return ScrapeTweetsResponse(
                response="No tweets found for the specified query",
                status="not_found",
                tweets_scraped=0
            )
        
        # Pass includes data to the storage function for better author information
        stored_count = await client.store_tweets_batch(tweets, ctx, includes)
        tweet_ids = [tweet.get('id', '') for tweet in tweets if tweet.get('id')]
        verified_count = await client.verify_tweets_stored(tweet_ids, ctx)
        
        return ScrapeTweetsResponse(
            response=f"Successfully scraped and stored {stored_count} tweets for query: '{msg.query}'. Found: {len(tweets)}, Stored: {stored_count}, Verified: {verified_count}. Enhanced with author data from {len(includes.get('users', []))} users.",
            status="success",
            tweets_scraped=stored_count
        )
    except Exception as e:
        ctx.logger.error(f"Error scraping tweets: {str(e)}")
        return ScrapeTweetsResponse(
            response=f"An error occurred: {str(e)}",
            status="error",
            tweets_scraped=0
        )

@agent.on_rest_post("/x/sentiment", SentimentRequest, SentimentResponse)
async def handle_sentiment_analysis(ctx: Context, msg: SentimentRequest) -> SentimentResponse:
    try:
        ctx.logger.info(f"Received sentiment analysis request for range: {msg.range}")
        
        # Calculate date range
        now = datetime.now(timezone.utc)
        if msg.range == "day":
            start_date = now - timedelta(days=1)
        elif msg.range == "week":
            start_date = now - timedelta(weeks=1)
        elif msg.range == "month":
            start_date = now - timedelta(days=30)
        else:
            return SentimentResponse(
                response="Invalid range. Use 'day', 'week', or 'month'",
                status="error"
            )
        
        # Query sentiment counts
        query = f"""
            SELECT 
                sentiment,
                COUNT(*) as count
            FROM vaccine_tweets 
            WHERE fetched_at >= '{start_date.isoformat()}'
            GROUP BY sentiment;
        """
        
        result = await client.execute_sql(query, ctx)
        
        # Check if we got results - parse the content to see if it's empty
        has_results = False
        if result and result.content:
            try:
                content = result.content[0].text if isinstance(result.content, list) else result.content
                data = json.loads(content)
                has_results = bool(data)  # True if data array is not empty
                ctx.logger.info(f"Initial query returned {len(data) if data else 0} sentiment groups")
            except Exception as e:
                ctx.logger.error(f"Error parsing initial query result: {str(e)}")
                has_results = False
        
        # If no results found in the specified range, try expanding the range
        if not has_results:
            ctx.logger.info(f"No data found for {msg.range} range, checking if any data exists...")
            
            # Check if there's any data at all
            check_query = "SELECT COUNT(*) as total FROM vaccine_tweets;"
            check_result = await client.execute_sql(check_query, ctx)
            
            if check_result and check_result.content:
                check_content = check_result.content[0].text if isinstance(check_result.content, list) else check_result.content
                check_data = json.loads(check_content)
                total_tweets = check_data[0].get('total', 0) if check_data else 0
                
                if total_tweets > 0:
                    ctx.logger.info(f"Found {total_tweets} total tweets, using all available data as fallback")
                    # Get all data without date filtering
                    fallback_query = """
                        SELECT 
                            sentiment,
                            COUNT(*) as count
                        FROM vaccine_tweets 
                        GROUP BY sentiment;
                    """
                    result = await client.execute_sql(fallback_query, ctx)
                    
                    # Add note about fallback in response
                    fallback_note = f" (showing all available data - {total_tweets} tweets)"
                else:
                    return SentimentResponse(
                        response="No data found in database",
                        status="not_found"
                    )
            else:
                return SentimentResponse(
                    response="Error checking database",
                    status="error"
                )
        else:
            fallback_note = ""
        
        # Parse results
        try:
            content = result.content[0].text if isinstance(result.content, list) else result.content
            data = json.loads(content)
            
            # Initialize counts
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            total_count = 0
            
            # Parse sentiment counts
            for row in data:
                sentiment = row.get('sentiment', 'neutral')
                count = row.get('count', 0)
                total_count += count
                
                if sentiment == 'positive':
                    positive_count = count
                elif sentiment == 'negative':
                    negative_count = count
                elif sentiment == 'neutral':
                    neutral_count = count
            
            return SentimentResponse(
                response=f"Sentiment analysis for {msg.range}: Positive: {positive_count}, Negative: {negative_count}, Neutral: {neutral_count}{fallback_note}",
                status="success",
                positive_count=positive_count,
                negative_count=negative_count,
                neutral_count=neutral_count,
                total_count=total_count
            )
            
        except Exception as e:
            ctx.logger.error(f"Error parsing sentiment results: {str(e)}")
            return SentimentResponse(
                response=f"Error parsing results: {str(e)}",
                status="error"
            )
        
    except Exception as e:
        ctx.logger.error(f"Error analyzing sentiment: {str(e)}")
        return SentimentResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_post("/x/trend", TrendRequest, TrendResponse)
async def handle_trend_analysis(ctx: Context, msg: TrendRequest) -> TrendResponse:
    try:
        ctx.logger.info(f"Received trend analysis request for range: {msg.range}")
        
        # Calculate date range
        now = datetime.now(timezone.utc)
        if msg.range == "day":
            start_date = now - timedelta(days=1)
        elif msg.range == "week":
            start_date = now - timedelta(weeks=1)
        elif msg.range == "month":
            start_date = now - timedelta(days=30)
        else:
            return TrendResponse(
                response="Invalid range. Use 'day', 'week', or 'month'",
                status="error"
            )
        
        # Query daily sentiment trends
        query = f"""
            SELECT 
                DATE(fetched_at) as date,
                sentiment,
                COUNT(*) as count
            FROM vaccine_tweets 
            WHERE fetched_at >= '{start_date.isoformat()}'
            GROUP BY DATE(fetched_at), sentiment
            ORDER BY date;
        """
        
        result = await client.execute_sql(query, ctx)
        
        if not result or not result.content:
            return TrendResponse(
                response="No data found for the specified range",
                status="not_found"
            )
        
        # Parse and format results
        try:
            content = result.content[0].text if isinstance(result.content, list) else result.content
            data = json.loads(content)
            
            # Group by date
            trends_by_date = {}
            for row in data:
                date = row.get('date')
                sentiment = row.get('sentiment')
                count = row.get('count', 0)
                
                if date not in trends_by_date:
                    trends_by_date[date] = {'positive': 0, 'negative': 0, 'neutral': 0}
                
                trends_by_date[date][sentiment] = count
            
            # Convert to list format for easier consumption
            trends_list = []
            for date, sentiments in trends_by_date.items():
                trends_list.append({
                    'date': date,
                    'positive': sentiments['positive'],
                    'negative': sentiments['negative'],
                    'neutral': sentiments['neutral'],
                    'total': sum(sentiments.values())
                })
            
            return TrendResponse(
                response=f"Trend analysis for {msg.range} completed",
                status="success",
                trends_data=json.dumps(trends_list, indent=2)
            )
            
        except Exception as e:
            ctx.logger.error(f"Error parsing trend results: {str(e)}")
            return TrendResponse(
                response=f"Error parsing results: {str(e)}",
                status="error"
            )
        
    except Exception as e:
        ctx.logger.error(f"Error analyzing trends: {str(e)}")
        return TrendResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_post("/x/tweets", TweetsRequest, TweetsResponse)
async def handle_get_tweets(ctx: Context, msg: TweetsRequest) -> TweetsResponse:
    try:
        ctx.logger.info(f"Received tweets request: range={msg.range}, sentiment={msg.sentiment}, limit={msg.limit}")
        
        # Calculate date range
        now = datetime.now(timezone.utc)
        if msg.range == "day":
            start_date = now - timedelta(days=1)
        elif msg.range == "week":
            start_date = now - timedelta(weeks=1)
        elif msg.range == "month":
            start_date = now - timedelta(days=30)
        else:
            return TweetsResponse(
                response="Invalid range. Use 'day', 'week', or 'month'",
                status="error"
            )
        
        # Build WHERE clause
        where_clauses = [f"fetched_at >= '{start_date.isoformat()}'"]
        if msg.sentiment:
            where_clauses.append(f"sentiment = '{msg.sentiment}'")
        where_sql = " AND ".join(where_clauses)
        
        # Query tweets
        query = f"""
            SELECT 
                tweet_id, content, author_username, author_name, author_profile_image,
                created_at, fetched_at, likes, retweets, replies,
                sentiment, impact_score, engagement_score
            FROM vaccine_tweets 
            WHERE {where_sql}
            ORDER BY fetched_at DESC
            LIMIT {msg.limit} OFFSET {msg.offset};
        """
        
        result = await client.execute_sql(query, ctx)
        
        if not result or not result.content:
            return TweetsResponse(
                response="No tweets found for the specified filters",
                status="not_found"
            )
        
        # Parse results
        try:
            content = result.content[0].text if isinstance(result.content, list) else result.content
            tweets_data = json.loads(content)
            
            # Get total count
            count_query = f"""
                SELECT COUNT(*) as total
                FROM vaccine_tweets 
                WHERE {where_sql};
            """
            count_result = await client.execute_sql(count_query, ctx)
            count_content = count_result.content[0].text if isinstance(count_result.content, list) else count_result.content
            count_data = json.loads(count_content)
            total_count = count_data[0].get('total', 0) if count_data else 0
            
            return TweetsResponse(
                response=f"Retrieved {len(tweets_data)} tweets",
                status="success",
                tweets=json.dumps(tweets_data, indent=2),
                total_count=total_count
            )
            
        except Exception as e:
            ctx.logger.error(f"Error parsing tweets results: {str(e)}")
            return TweetsResponse(
                response=f"Error parsing results: {str(e)}",
                status="error"
            )
        
    except Exception as e:
        ctx.logger.error(f"Error retrieving tweets: {str(e)}")
        return TweetsResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_post("/x/user_analysis", UserAnalysisRequest, UserAnalysisResponse)
async def handle_user_analysis(ctx: Context, msg: UserAnalysisRequest) -> UserAnalysisResponse:
    try:
        range_info = f" (range: {msg.range})" if msg.range else ""
        ctx.logger.info(f"Received user analysis request for: {msg.username}{range_info}")
        
        # Calculate date range if specified
        date_filter = ""
        if msg.range:
            now = datetime.now(timezone.utc)
            if msg.range == "day":
                start_date = now - timedelta(days=1)
            elif msg.range == "week":
                start_date = now - timedelta(weeks=1)
            elif msg.range == "month":
                start_date = now - timedelta(days=30)
            else:
                start_date = None
            
            if start_date:
                date_filter = f" AND fetched_at >= '{start_date.isoformat()}'"
        
        # Query user's tweets with optional date filtering
        query = f"""
            SELECT 
                author_username,
                author_name,
                COUNT(*) as total_tweets,
                AVG(impact_score) as avg_impact_score,
                AVG(engagement_score) as avg_engagement_score,
                sentiment,
                COUNT(*) as sentiment_count
            FROM vaccine_tweets 
            WHERE author_username = '{msg.username}'{date_filter}
            GROUP BY author_username, author_name, sentiment
            ORDER BY sentiment_count DESC;
        """
        
        result = await client.execute_sql(query, ctx)
        
        if not result or not result.content:
            return UserAnalysisResponse(
                response="No data found for the specified user",
                status="not_found"
            )
        
        # Parse results
        try:
            content = result.content[0].text if isinstance(result.content, list) else result.content
            data = json.loads(content)
            
            # Process user analysis data
            user_analysis = {
                'username': msg.username,
                'total_tweets': 0,
                'avg_impact_score': 0,
                'avg_engagement_score': 0,
                'sentiment_breakdown': {},
                'top_tweets': []
            }
            
            total_tweets = 0
            total_impact = 0
            total_engagement = 0
            
            for row in data:
                count = row.get('sentiment_count', 0)
                sentiment = row.get('sentiment', 'neutral')
                impact = row.get('avg_impact_score', 0)
                engagement = row.get('avg_engagement_score', 0)
                
                total_tweets += count
                total_impact += impact * count
                total_engagement += engagement * count
                user_analysis['sentiment_breakdown'][sentiment] = count
            
            if total_tweets > 0:
                user_analysis['total_tweets'] = total_tweets
                user_analysis['avg_impact_score'] = total_impact / total_tweets
                user_analysis['avg_engagement_score'] = total_engagement / total_tweets
            
            # Get top tweets by impact score with same date filter
            top_tweets_query = f"""
                SELECT content, impact_score, sentiment, created_at
                FROM vaccine_tweets 
                WHERE author_username = '{msg.username}'{date_filter}
                ORDER BY impact_score DESC
                LIMIT 5;
            """
            
            top_tweets_result = await client.execute_sql(top_tweets_query, ctx)
            if top_tweets_result and top_tweets_result.content:
                top_content = top_tweets_result.content[0].text if isinstance(top_tweets_result.content, list) else top_tweets_result.content
                top_tweets_data = json.loads(top_content)
                user_analysis['top_tweets'] = top_tweets_data
            
            return UserAnalysisResponse(
                response=f"User analysis completed for {msg.username}",
                status="success",
                user_data=json.dumps(user_analysis, indent=2)
            )
            
        except Exception as e:
            ctx.logger.error(f"Error parsing user analysis results: {str(e)}")
            return UserAnalysisResponse(
                response=f"Error parsing results: {str(e)}",
                status="error"
            )
        
    except Exception as e:
        ctx.logger.error(f"Error analyzing user: {str(e)}")
        return UserAnalysisResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_post("/x/trending_topics", TrendingTopicsRequest, TrendingTopicsResponse)
async def handle_trending_topics(ctx: Context, msg: TrendingTopicsRequest) -> TrendingTopicsResponse:
    try:
        ctx.logger.info("Received trending topics request")
        try:
            rapidapi_response = rapidapi_search_tweets("vaccine", 50)
            tweets_data = parse_rapidapi_response(rapidapi_response)
        except Exception as e:
            ctx.logger.error(f"RapidAPI error: {str(e)}")
            return TrendingTopicsResponse(
                response=f"RapidAPI error: {str(e)}",
                status="error"
            )
        tweets = tweets_data.get('data', [])
        if not tweets:
            return TrendingTopicsResponse(
                response="No trending topics found",
                status="not_found"
            )
        hashtags = {}
        keywords = {}
        for tweet in tweets:
            text = tweet.get('text', '')
            hashtag_matches = re.findall(r'#(\w+)', text.lower())
            for hashtag in hashtag_matches:
                hashtags[hashtag] = hashtags.get(hashtag, 0) + 1
            words = re.findall(r'\b\w+\b', text.lower())
            for word in words:
                if len(word) > 3 and word not in ['vaccine', 'vaccination', 'covid', 'the', 'and', 'for', 'with', 'this', 'that', 'have', 'been', 'they', 'will', 'from', 'their', 'said', 'each', 'which', 'there', 'were', 'time', 'would', 'could', 'about', 'into', 'more', 'your', 'what', 'some', 'very', 'when', 'just', 'know', 'take', 'than', 'them', 'well', 'only', 'come', 'over', 'think', 'also', 'back', 'after', 'work', 'first', 'good', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us']:
                    keywords[word] = keywords.get(word, 0) + 1
        trending_hashtags = sorted(hashtags.items(), key=lambda x: x[1], reverse=True)[:10]
        trending_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:10]
        trending_data = {
            'hashtags': [{'tag': tag, 'count': count} for tag, count in trending_hashtags],
            'keywords': [{'word': word, 'count': count} for word, count in trending_keywords]
        }
        return TrendingTopicsResponse(
            response="Trending topics analysis completed",
            status="success",
            topics=json.dumps(trending_data, indent=2)
        )
    except Exception as e:
        ctx.logger.error(f"Error getting trending topics: {str(e)}")
        return TrendingTopicsResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_post("/x/scrape_multiple", ScrapeMultipleQueriesRequest, ScrapeMultipleQueriesResponse)
async def handle_scrape_multiple_queries(ctx: Context, msg: ScrapeMultipleQueriesRequest) -> ScrapeMultipleQueriesResponse:
    try:
        ctx.logger.info(f"Received multiple query scraping request")
        
        # Define balanced search queries for diverse sentiment
        default_queries = [
            "vaccine safety",           # More neutral/informational
            "vaccination benefits",     # Likely positive
            "vaccine side effects",     # Mixed sentiment
            "covid vaccine",           # Mixed/controversial
            "flu vaccine",             # Generally more positive
            "vaccine hesitancy",       # Likely negative
            "immunization program",    # More neutral/official
            "vaccine research",        # Neutral/scientific
            "vaccine mandate",         # Controversial/mixed
            "vaccine breakthrough"     # Mixed sentiment
        ]
        
        queries = msg.queries if msg.queries else default_queries
        max_results_per_query = msg.tweets_per_query
        
        total_scraped = 0
        query_breakdown = {}
        
        for query in queries:
            try:
                ctx.logger.info(f"Scraping query: {query}")
                rapidapi_response = rapidapi_search_tweets(query, max_results_per_query)
                tweets_data = parse_rapidapi_response(rapidapi_response)
                tweets = tweets_data.get('data', [])
                includes = tweets_data.get('includes', {})
                
                if tweets:
                    stored_count = await client.store_tweets_batch(tweets, ctx, includes)
                    total_scraped += stored_count
                    query_breakdown[query] = {
                        "found": len(tweets),
                        "stored": stored_count
                    }
                    ctx.logger.info(f"Query '{query}': found {len(tweets)}, stored {stored_count}")
                else:
                    query_breakdown[query] = {"found": 0, "stored": 0}
                    
            except Exception as e:
                ctx.logger.error(f"Error scraping query '{query}': {str(e)}")
                query_breakdown[query] = {"found": 0, "stored": 0, "error": str(e)}
        
        return ScrapeMultipleQueriesResponse(
            response=f"Successfully scraped {total_scraped} tweets from {len(queries)} queries",
            status="success",
            total_tweets_scraped=total_scraped,
            query_results=json.dumps(query_breakdown, indent=2)
        )
        
    except Exception as e:
        ctx.logger.error(f"Error in multiple query scraping: {str(e)}")
        return ScrapeMultipleQueriesResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_post("/x/dashboard", DashboardDataRequest, DashboardDataResponse)
async def handle_dashboard_data(ctx: Context, msg: DashboardDataRequest) -> DashboardDataResponse:
    try:
        ctx.logger.info(f"Received dashboard data request for range: {msg.range}")
        
        # Calculate date range
        now = datetime.now(timezone.utc)
        if msg.range == "day":
            start_date = now - timedelta(days=1)
        elif msg.range == "week":
            start_date = now - timedelta(weeks=1)
        elif msg.range == "month":
            start_date = now - timedelta(days=30)
        else:
            start_date = now - timedelta(weeks=1)  # Default to week
        
        # Get sentiment summary
        sentiment_query = f"""
            SELECT 
                sentiment,
                COUNT(*) as count,
                ROUND(AVG(impact_score)::numeric, 2) as avg_impact,
                ROUND(AVG(engagement_score)::numeric, 2) as avg_engagement
            FROM vaccine_tweets 
            WHERE fetched_at >= '{start_date.isoformat()}'
            GROUP BY sentiment;
        """
        sentiment_result = await client.execute_sql(sentiment_query, ctx)
        sentiment_data = json.loads(sentiment_result.content[0].text if isinstance(sentiment_result.content, list) else sentiment_result.content) if sentiment_result.content else []
        
        # Get top tweets by engagement
        top_tweets_query = f"""
            SELECT 
                tweet_id, content, author_username, author_name,
                likes, retweets, replies, sentiment, impact_score, engagement_score, created_at
            FROM vaccine_tweets 
            WHERE fetched_at >= '{start_date.isoformat()}'
            ORDER BY impact_score DESC
            LIMIT 10;
        """
        top_tweets_result = await client.execute_sql(top_tweets_query, ctx)
        top_tweets_data = json.loads(top_tweets_result.content[0].text if isinstance(top_tweets_result.content, list) else top_tweets_result.content) if top_tweets_result.content else []
        
        # Get user insights
        user_insights_query = f"""
            SELECT 
                author_username, author_name,
                COUNT(*) as tweet_count,
                SUM(likes) as total_likes,
                SUM(retweets) as total_retweets,
                ROUND(AVG(impact_score)::numeric, 2) as avg_impact,
                sentiment
            FROM vaccine_tweets 
            WHERE fetched_at >= '{start_date.isoformat()}'
            GROUP BY author_username, author_name, sentiment
            ORDER BY total_likes + total_retweets DESC
            LIMIT 15;
        """
        user_insights_result = await client.execute_sql(user_insights_query, ctx)
        user_insights_data = json.loads(user_insights_result.content[0].text if isinstance(user_insights_result.content, list) else user_insights_result.content) if user_insights_result.content else []
        
        # Get trending topics (simplified)
        trending_query = f"""
            SELECT content FROM vaccine_tweets 
            WHERE fetched_at >= '{start_date.isoformat()}'
            ORDER BY impact_score DESC
            LIMIT 50;
        """
        trending_result = await client.execute_sql(trending_query, ctx)
        trending_content = json.loads(trending_result.content[0].text if isinstance(trending_result.content, list) else trending_result.content) if trending_result.content else []
        
        # Extract hashtags and keywords
        hashtags = {}
        keywords = {}
        for row in trending_content:
            content = row.get('content', '')
            # Extract hashtags
            hashtag_matches = re.findall(r'#(\w+)', content.lower())
            for hashtag in hashtag_matches:
                hashtags[hashtag] = hashtags.get(hashtag, 0) + 1
            
            # Extract keywords
            words = re.findall(r'\b\w+\b', content.lower())
            for word in words:
                if len(word) > 4 and word not in ['vaccine', 'vaccination', 'covid', 'coronavirus']:
                    keywords[word] = keywords.get(word, 0) + 1
        
        trending_topics = {
            'hashtags': sorted(hashtags.items(), key=lambda x: x[1], reverse=True)[:10],
            'keywords': sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:10]
        }
        
        return DashboardDataResponse(
            response="Dashboard data compiled successfully",
            status="success",
            sentiment_summary=json.dumps(sentiment_data, indent=2),
            top_tweets=json.dumps(top_tweets_data, indent=2),
            trending_topics=json.dumps(trending_topics, indent=2),
            user_insights=json.dumps(user_insights_data, indent=2)
        )
        
    except Exception as e:
        ctx.logger.error(f"Error getting dashboard data: {str(e)}")
        return DashboardDataResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_post("/x/sentiment_timeline", SentimentTimelineRequest, SentimentTimelineResponse)
async def handle_sentiment_timeline(ctx: Context, msg: SentimentTimelineRequest) -> SentimentTimelineResponse:
    try:
        ctx.logger.info(f"Received sentiment timeline request: range={msg.range}, granularity={msg.granularity}")
        
        # Calculate date range
        now = datetime.now(timezone.utc)
        if msg.range == "day":
            start_date = now - timedelta(days=1)
            date_trunc = "hour" if msg.granularity == "hour" else "day"
        elif msg.range == "week":
            start_date = now - timedelta(weeks=1)
            date_trunc = "day"
        elif msg.range == "month":
            start_date = now - timedelta(days=30)
            date_trunc = "day"
        else:
            start_date = now - timedelta(weeks=1)
            date_trunc = "day"
        
        # Get timeline data
        timeline_query = f"""
            SELECT 
                DATE_TRUNC('{date_trunc}', fetched_at) as time_bucket,
                sentiment,
                COUNT(*) as count,
                ROUND(AVG(impact_score)::numeric, 2) as avg_impact
            FROM vaccine_tweets 
            WHERE fetched_at >= '{start_date.isoformat()}'
            GROUP BY DATE_TRUNC('{date_trunc}', fetched_at), sentiment
            ORDER BY time_bucket, sentiment;
        """
        
        result = await client.execute_sql(timeline_query, ctx)
        timeline_data = json.loads(result.content[0].text if isinstance(result.content, list) else result.content) if result.content else []
        
        return SentimentTimelineResponse(
            response="Sentiment timeline data compiled successfully",
            status="success",
            timeline_data=json.dumps(timeline_data, indent=2)
        )
        
    except Exception as e:
        ctx.logger.error(f"Error getting sentiment timeline: {str(e)}")
        return SentimentTimelineResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_post("/x/top_users", TopUsersRequest, TopUsersResponse)
async def handle_top_users(ctx: Context, msg: TopUsersRequest) -> TopUsersResponse:
    try:
        ctx.logger.info(f"Received top users request: range={msg.range}, metric={msg.metric}")
        
        # Calculate date range
        now = datetime.now(timezone.utc)
        if msg.range == "day":
            start_date = now - timedelta(days=1)
        elif msg.range == "week":
            start_date = now - timedelta(weeks=1)
        elif msg.range == "month":
            start_date = now - timedelta(days=30)
        else:
            start_date = now - timedelta(weeks=1)
        
        # Build query based on metric
        if msg.metric == "engagement":
            order_by = "total_engagement DESC"
            metric_calc = "SUM(likes + retweets + replies) as total_engagement"
        elif msg.metric == "impact":
            order_by = "avg_impact DESC"
            metric_calc = "ROUND(AVG(impact_score)::numeric, 2) as avg_impact"
        elif msg.metric == "tweets":
            order_by = "tweet_count DESC"
            metric_calc = "COUNT(*) as tweet_count"
        else:
            order_by = "total_engagement DESC"
            metric_calc = "SUM(likes + retweets + replies) as total_engagement"
        
        users_query = f"""
            SELECT 
                author_username, author_name,
                COUNT(*) as tweet_count,
                SUM(likes) as total_likes,
                SUM(retweets) as total_retweets,
                SUM(replies) as total_replies,
                {metric_calc},
                ROUND(AVG(impact_score)::numeric, 2) as avg_impact_score,
                ROUND(AVG(engagement_score)::numeric, 2) as avg_engagement_score,
                MODE() WITHIN GROUP (ORDER BY sentiment) as dominant_sentiment
            FROM vaccine_tweets 
            WHERE fetched_at >= '{start_date.isoformat()}'
            GROUP BY author_username, author_name
            HAVING COUNT(*) >= 2
            ORDER BY {order_by}
            LIMIT 20;
        """
        
        result = await client.execute_sql(users_query, ctx)
        users_data = json.loads(result.content[0].text if isinstance(result.content, list) else result.content) if result.content else []
        
        return TopUsersResponse(
            response=f"Top users by {msg.metric} compiled successfully",
            status="success",
            users_data=json.dumps(users_data, indent=2)
        )
        
    except Exception as e:
        ctx.logger.error(f"Error getting top users: {str(e)}")
        return TopUsersResponse(
            response=f"An error occurred: {str(e)}",
            status="error"
        )

@agent.on_rest_get("/x/debug/tweets_date_range", DebugResponse)
async def handle_debug_tweets_date_range(ctx: Context) -> DebugResponse:
    try:
        ctx.logger.info("Received debug request for tweet date range")
        
        # Get date range and count
        query = """
            SELECT 
                MIN(fetched_at) as min_date, 
                MAX(fetched_at) as max_date, 
                COUNT(*) as total_count
            FROM vaccine_tweets;
        """
        result = await client.execute_sql(query, ctx)
        
        if result and result.content:
            content = result.content[0].text if isinstance(result.content, list) else result.content
            data = json.loads(content)
            min_date = data[0].get('min_date') if data else None
            max_date = data[0].get('max_date') if data else None
            total_count = data[0].get('total_count') if data else 0
            
            return DebugResponse(
                response=f"Database contains {total_count} tweets from {min_date} to {max_date}",
                status="success",
                data=json.dumps(data, indent=2)
            )
        else:
            return DebugResponse(
                response="No tweets found in database",
                status="not_found",
                data="[]"
            )
    except Exception as e:
        ctx.logger.error(f"Error getting tweet date range: {str(e)}")
        return DebugResponse(
            response=f"An error occurred: {str(e)}",
            status="error",
            data=""
        )

if __name__ == "__main__":
    agent.run()
