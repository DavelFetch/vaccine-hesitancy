from uagents import Model
from typing import Optional, List
from datetime import datetime

# Tweet scraping models
class ScrapeTweetsRequest(Model):
    query: str = "vaccine"  # Search query for tweets (only required parameter)
    range: Optional[str] = None  # day, week, month - for time-based scraping

class ScrapeTweetsResponse(Model):
    response: str
    status: str
    tweets_scraped: int = 0

# Multiple query scraping for balanced data
class ScrapeMultipleQueriesRequest(Model):
    queries: Optional[List[str]] = None  # If None, uses default balanced queries
    tweets_per_query: int = 10
    range: Optional[str] = None  # day, week, month

class ScrapeMultipleQueriesResponse(Model):
    response: str
    status: str
    total_tweets_scraped: int = 0
    query_results: str = ""  # JSON string with per-query breakdown

# Sentiment analysis models
class SentimentRequest(Model):
    range: str = "day"  # day, week, month

class SentimentResponse(Model):
    response: str
    status: str
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    total_count: int = 0

# Trend analysis models
class TrendRequest(Model):
    range: str = "day"  # day, week, month

class TrendResponse(Model):
    response: str
    status: str
    trends_data: str = ""  # JSON string with daily sentiment trends

# Tweet retrieval models
class TweetsRequest(Model):
    range: str = "day"  # day, week, month
    sentiment: Optional[str] = None  # positive, negative, neutral
    limit: int = 50
    offset: int = 0

class TweetsResponse(Model):
    response: str
    status: str
    tweets: str = ""  # JSON string with tweet data
    total_count: int = 0

# Tweet data model for storage
class TweetData(Model):
    tweet_id: str
    content: str
    author_username: str
    author_name: str
    author_profile_image: Optional[str] = None
    created_at: datetime
    fetched_at: datetime
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    sentiment: str = "neutral"  # positive, negative, neutral
    impact_score: float = 0.0
    engagement_score: float = 0.0

# User analysis models with time range
class UserAnalysisRequest(Model):
    username: str
    range: Optional[str] = None  # day, week, month, all

class UserAnalysisResponse(Model):
    response: str
    status: str
    user_data: str = ""  # JSON string with user analysis data

# Trending topics models
class TrendingTopicsRequest(Model):
    pass

class TrendingTopicsResponse(Model):
    response: str
    status: str
    topics: str = ""  # JSON string with trending topics

# Frontend-friendly models
class DashboardDataRequest(Model):
    range: str = "week"  # day, week, month

class DashboardDataResponse(Model):
    response: str
    status: str
    sentiment_summary: str = ""  # JSON with sentiment breakdown
    top_tweets: str = ""  # JSON with highest engagement tweets
    trending_topics: str = ""  # JSON with trending hashtags/keywords
    user_insights: str = ""  # JSON with top users by engagement

class SentimentTimelineRequest(Model):
    range: str = "week"  # day, week, month
    granularity: str = "day"  # hour, day

class SentimentTimelineResponse(Model):
    response: str
    status: str
    timeline_data: str = ""  # JSON with time-series sentiment data

class TopUsersRequest(Model):
    range: str = "week"  # day, week, month
    metric: str = "engagement"  # engagement, impact, tweets

class TopUsersResponse(Model):
    response: str
    status: str
    users_data: str = ""  # JSON with top users data 