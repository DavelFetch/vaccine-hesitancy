// Common types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

// Page types
export type PageType = 'health-board' | 'social-media' | 'vaccine-resources' | 'voice-analysis';

// Health Board / Vaccine Hesitancy Types
export interface VHRegionRequest {
  region: string;
}

export interface VHRegionResponse {
  response: string;
  status: string;
}

export interface VHSexRequest {
  sex: string;
}

export interface VHSexResponse {
  response: string;
  status: string;
}

export interface VHEthnicityRequest {
  ethnicity: string;
}

export interface VHEthnicityResponse {
  response: string;
  status: string;
}

export interface VHAgentRequest {
  query: string;
}

export interface VHAgentResponse {
  response: string;
  status: string;
}

// X Analysis / Social Media Types
export interface SentimentRequest {
  range: string; // day, week, month
}

export interface SentimentResponse {
  response: string;
  status: string;
  positive_count: number;
  negative_count: number;
  neutral_count: number;
  total_count: number;
}

export interface TweetData {
  tweet_id: string;
  content: string;
  author_username: string;
  author_name: string;
  author_profile_image?: string;
  created_at: string;
  fetched_at: string;
  likes: number;
  retweets: number;
  replies: number;
  sentiment: string;
  impact_score: number;
  engagement_score: number;
}

// Additional X Analysis Types
export interface TweetsRequest {
  range: string; // day, week, month
  sentiment?: string; // positive, negative, neutral
  limit: number;
  offset: number;
}

export interface TweetsResponse {
  response: string;
  status: string;
  tweets: string; // JSON string of TweetData[]
  total_count: number;
}

export interface DashboardDataRequest {
  range: string; // day, week, month
}

export interface DashboardDataResponse {
  response: string;
  status: string;
  sentiment_summary: string; // JSON string
  top_tweets: string; // JSON string
  trending_topics: string; // JSON string
  user_insights: string; // JSON string
}

export interface SentimentTimelineRequest {
  range: string; // day, week, month
  granularity?: string; // hour, day
}

export interface SentimentTimelineResponse {
  response: string;
  status: string;
  timeline_data: string; // JSON string
}

export interface TopUsersRequest {
  range: string; // day, week, month
  metric: string; // engagement, impact, tweets
}

export interface TopUsersResponse {
  response: string;
  status: string;
  users_data: string; // JSON string
}

export interface TrendingTopicsRequest {
  // No parameters needed for trending topics request
}

export interface TrendingTopicsResponse {
  response: string;
  status: string;
  topics: string; // JSON string
}

// Voice Analysis Types
export interface AudioUploadRequest {
  audio_data: string; // base64 encoded
  filename: string;
  content_type: string;
  analysis_options: {
    hesitancy_analysis: boolean;
    keyword_extraction: boolean;
  };
}

export interface AudioAnalysisResponse {
  request_id: string;
  success: boolean;
  transcript: string;
  transcript_confidence: number;
  hesitancy_analysis: Record<string, any>;
  keywords: string[];
  processing_time: number;
  audio_metadata: Record<string, any>;
  timestamp: string;
}

export interface TextAnalysisRequest {
  text: string;
  analysis_options: {
    hesitancy_analysis: boolean;
    keyword_extraction: boolean;
    sentiment_analysis: boolean;
  };
}

export interface TextAnalysisResponse {
  request_id: string;
  success: boolean;
  hesitancy_analysis: Record<string, any>;
  keywords: string[];
  processing_time: number;
  timestamp: string;
}

// Vaccine Resource Types
export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

// UI State Types
export interface FilterState {
  region?: string;
  sex?: string;
  ethnicity?: string;
  ageGroup?: string;
  religion?: string;
  timeRange?: string;
}

export interface ComparisonState {
  leftPanel: FilterState;
  rightPanel: FilterState;
  comparisonMode: boolean;
} 