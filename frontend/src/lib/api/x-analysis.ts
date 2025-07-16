import { BaseApiClient } from './base';
import { config } from '@/lib/config';
import { ApiResponse } from '@/types';
import {
  SentimentRequest,
  SentimentResponse,
  TweetsRequest,
  TweetsResponse,
  DashboardDataRequest,
  DashboardDataResponse,
  SentimentTimelineRequest,
  SentimentTimelineResponse,
  TopUsersRequest,
  TopUsersResponse,
  TrendingTopicsRequest,
  TrendingTopicsResponse,
  TweetData
} from '@/types';

export class XAnalysisApi extends BaseApiClient {
  constructor() {
    super(config.agents.xAnalysis);
  }

  // Sentiment Analysis
  async getSentimentAnalysis(request: SentimentRequest): Promise<ApiResponse<SentimentResponse>> {
    return this.post<SentimentResponse>('/x/sentiment', request);
  }

  // Get Tweets
  async getTweets(request: TweetsRequest): Promise<ApiResponse<TweetsResponse>> {
    return this.post<TweetsResponse>('/x/tweets', request);
  }

  // Dashboard Data (comprehensive overview)
  async getDashboardData(request: DashboardDataRequest): Promise<ApiResponse<DashboardDataResponse>> {
    return this.post<DashboardDataResponse>('/x/dashboard', request);
  }

  // Sentiment Timeline
  async getSentimentTimeline(request: SentimentTimelineRequest): Promise<ApiResponse<SentimentTimelineResponse>> {
    return this.post<SentimentTimelineResponse>('/x/sentiment_timeline', request);
  }

  // Top Users
  async getTopUsers(request: TopUsersRequest): Promise<ApiResponse<TopUsersResponse>> {
    return this.post<TopUsersResponse>('/x/top_users', request);
  }

  // Trending Topics
  async getTrendingTopics(request: TrendingTopicsRequest): Promise<ApiResponse<TrendingTopicsResponse>> {
    return this.post<TrendingTopicsResponse>('/x/trending_topics', request);
  }

  // Scrape Tweets (new)
  async scrapeTweets(query: string, range?: string): Promise<any> {
    return this.post('/x/scrape', { query, range });
  }

  // Convenience methods for common use cases
  async getWeeklySentiment(): Promise<ApiResponse<SentimentResponse>> {
    return this.getSentimentAnalysis({ range: 'week' });
  }

  async getDailySentiment(): Promise<ApiResponse<SentimentResponse>> {
    return this.getSentimentAnalysis({ range: 'day' });
  }

  async getMonthlySentiment(): Promise<ApiResponse<SentimentResponse>> {
    return this.getSentimentAnalysis({ range: 'month' });
  }

  async getRecentTweets(sentiment?: string, limit: number = 20): Promise<ApiResponse<TweetsResponse>> {
    return this.getTweets({
      range: 'week',
      sentiment,
      limit,
      offset: 0
    });
  }

  async getWeeklyDashboard(): Promise<ApiResponse<DashboardDataResponse>> {
    return this.getDashboardData({ range: 'week' });
  }

  async getTopEngagementUsers(range: string = 'week'): Promise<ApiResponse<TopUsersResponse>> {
    return this.getTopUsers({
      range,
      metric: 'engagement'
    });
  }

  async getTopImpactUsers(range: string = 'week'): Promise<ApiResponse<TopUsersResponse>> {
    return this.getTopUsers({
      range,
      metric: 'impact'
    });
  }

  // Helper methods to parse JSON string responses
  parseTweetsData(tweetsResponse: TweetsResponse): TweetData[] {
    try {
      return JSON.parse(tweetsResponse.tweets);
    } catch (error) {
      console.error('Error parsing tweets data:', error);
      return [];
    }
  }

  parseSentimentSummary(dashboardResponse: DashboardDataResponse): any[] {
    try {
      return JSON.parse(dashboardResponse.sentiment_summary);
    } catch (error) {
      console.error('Error parsing sentiment summary:', error);
      return [];
    }
  }

  parseTopTweets(dashboardResponse: DashboardDataResponse): any[] {
    try {
      return JSON.parse(dashboardResponse.top_tweets);
    } catch (error) {
      console.error('Error parsing top tweets:', error);
      return [];
    }
  }

  parseTrendingTopics(dashboardResponse: DashboardDataResponse): any {
    try {
      return JSON.parse(dashboardResponse.trending_topics);
    } catch (error) {
      console.error('Error parsing trending topics:', error);
      return { hashtags: [], keywords: [] };
    }
  }

  parseUserInsights(dashboardResponse: DashboardDataResponse): any[] {
    try {
      return JSON.parse(dashboardResponse.user_insights);
    } catch (error) {
      console.error('Error parsing user insights:', error);
      return [];
    }
  }

  parseTimelineData(timelineResponse: SentimentTimelineResponse): any[] {
    try {
      return JSON.parse(timelineResponse.timeline_data);
    } catch (error) {
      console.error('Error parsing timeline data:', error);
      return [];
    }
  }

  parseUsersData(usersResponse: TopUsersResponse): any[] {
    try {
      return JSON.parse(usersResponse.users_data);
    } catch (error) {
      console.error('Error parsing users data:', error);
      return [];
    }
  }

  parseTopicsData(topicsResponse: TrendingTopicsResponse): any {
    try {
      return JSON.parse(topicsResponse.topics);
    } catch (error) {
      console.error('Error parsing topics data:', error);
      return { hashtags: [], keywords: [] };
    }
  }
}

// Create and export a singleton instance
export const xAnalysisApi = new XAnalysisApi(); 