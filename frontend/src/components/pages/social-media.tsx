'use client';

import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { xAnalysisApi } from '@/lib/api/x-analysis';
import { TweetData } from '@/types';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';

export function SocialMediaPage() {
  const [timeRange, setTimeRange] = useState('month'); // Changed from 'week' to 'month'
  const [sentimentFilter, setSentimentFilter] = useState('');
  const [scrapeQuery, setScrapeQuery] = useState('vaccine');
  const [scrapeLoading, setScrapeLoading] = useState(false);
  const [scrapeMsg, setScrapeMsg] = useState<string | null>(null);

  // Fetch sentiment data
  const { data: sentimentData, isLoading: sentimentLoading, error: sentimentError } = useQuery({
    queryKey: ['sentiment', timeRange],
    queryFn: () => xAnalysisApi.getSentimentAnalysis({ range: timeRange }),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Fetch dashboard data
  const { data: dashboardData, isLoading: dashboardLoading, error: dashboardError } = useQuery({
    queryKey: ['dashboard', timeRange],
    queryFn: () => xAnalysisApi.getDashboardData({ range: timeRange }),
    refetchInterval: 30000,
  });

  // Fetch recent tweets
  const { data: tweetsData, isLoading: tweetsLoading, error: tweetsError } = useQuery({
    queryKey: ['tweets', timeRange, sentimentFilter],
    queryFn: () => xAnalysisApi.getTweets({
      range: timeRange,
      sentiment: sentimentFilter || undefined,
      limit: 10,
      offset: 0
    }),
    refetchInterval: 30000,
  });

  // Parse dashboard data
  const parsedDashboardData = useMemo(() => {
    if (!dashboardData?.success || !dashboardData.data) return null;
    
    return {
      sentimentSummary: xAnalysisApi.parseSentimentSummary(dashboardData.data),
      topTweets: xAnalysisApi.parseTopTweets(dashboardData.data),
      trendingTopics: xAnalysisApi.parseTrendingTopics(dashboardData.data),
      userInsights: xAnalysisApi.parseUserInsights(dashboardData.data),
    };
  }, [dashboardData]);

  // Parse tweets data
  const parsedTweets = useMemo(() => {
    if (!tweetsData?.success || !tweetsData.data) return [];
    return xAnalysisApi.parseTweetsData(tweetsData.data);
  }, [tweetsData]);

  // Calculate sentiment percentages
  const sentimentStats = useMemo(() => {
    if (!sentimentData?.success || !sentimentData.data) {
      return { positive: 0, negative: 0, neutral: 0, total: 0 };
    }

    const data = sentimentData.data;
    const total = data.total_count || 0;
    
    if (total === 0) {
      return { positive: 0, negative: 0, neutral: 0, total: 0 };
    }

    return {
      positive: Math.round((data.positive_count / total) * 100),
      negative: Math.round((data.negative_count / total) * 100),
      neutral: Math.round((data.neutral_count / total) * 100),
      total: total
    };
  }, [sentimentData]);

  const isLoading = sentimentLoading || dashboardLoading || tweetsLoading;
  const hasError = sentimentError || dashboardError || tweetsError;

  const formatTimeAgo = (dateString: string) => {
    try {
      const date = new Date(dateString);
      const now = new Date();
      const diffInMinutes = Math.floor((now.getTime() - date.getTime()) / (1000 * 60));
      
      if (diffInMinutes < 60) {
        return `${diffInMinutes}m ago`;
      } else if (diffInMinutes < 1440) {
        return `${Math.floor(diffInMinutes / 60)}h ago`;
      } else {
        return `${Math.floor(diffInMinutes / 1440)}d ago`;
      }
    } catch {
      return 'Unknown';
    }
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment.toLowerCase()) {
      case 'positive': return 'bg-green-100 text-green-800';
      case 'negative': return 'bg-red-100 text-red-800';
      case 'neutral': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getAuthorInitials = (username: string, name: string) => {
    if (name && name !== username) {
      return name.split(' ').map(n => n[0]).join('').slice(0, 2).toUpperCase();
    }
    return username.slice(0, 2).toUpperCase();
  };

  const generateAvatarColor = (username: string) => {
    const colors = [
      'bg-blue-500', 'bg-green-500', 'bg-purple-500', 'bg-pink-500', 
      'bg-indigo-500', 'bg-red-500', 'bg-yellow-500', 'bg-teal-500'
    ];
    const index = username.length % colors.length;
    return colors[index];
  };

  const sentimentPieData = [
    { name: 'Positive', value: sentimentStats.positive, color: '#22c55e' }, // green-500
    { name: 'Negative', value: sentimentStats.negative, color: '#ef4444' }, // red-500
    { name: 'Neutral', value: sentimentStats.neutral, color: '#eab308' },   // yellow-500
  ];

  const handleScrape = async () => {
    setScrapeLoading(true);
    setScrapeMsg(null);
    try {
      const resp = await xAnalysisApi.scrapeTweets(scrapeQuery, timeRange);
      if (resp && resp.status === 'success') {
        setScrapeMsg(`✅ ${resp.response}`);
      } else {
        setScrapeMsg(`❌ ${resp?.response || 'Scraping failed.'}`);
      }
    } catch (e: any) {
      setScrapeMsg(`❌ ${e.message || 'Scraping failed.'}`);
    }
    setScrapeLoading(false);
  };

  if (hasError) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-3xl font-bold text-gray-900">Social Media Analysis</h2>
        </div>
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="text-red-800">
            <h3 className="font-medium">Error loading social media data</h3>
            <p className="text-sm mt-1">
              Please check that the X Analysis backend is running on http://localhost:8001
            </p>
            <p className="text-xs mt-2 text-red-600">
              Error details: {sentimentError?.message || dashboardError?.message || tweetsError?.message}
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold text-gray-900">Social Media Analysis</h2>
        <div className="text-sm text-gray-500">
          Twitter/X sentiment analysis and author impact scoring
        </div>
      </div>

      {/* Scraping Button */}
      <div className="bg-white p-4 rounded-lg shadow flex items-center space-x-4">
        <input
          type="text"
          className="border border-gray-300 rounded-md px-3 py-1 text-sm flex-1"
          placeholder="Enter search query (e.g. vaccine)"
          value={scrapeQuery}
          onChange={e => setScrapeQuery(e.target.value)}
          disabled={scrapeLoading}
        />
        <button
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 text-sm disabled:opacity-50"
          onClick={handleScrape}
          disabled={scrapeLoading || !scrapeQuery.trim()}
        >
          {scrapeLoading ? 'Scraping...' : 'Scrape Tweets'}
        </button>
        {scrapeMsg && (
          <span className={`ml-4 text-sm ${scrapeMsg.startsWith('✅') ? 'text-green-600' : 'text-red-600'}`}>{scrapeMsg}</span>
        )}
      </div>

      {/* Time Range Selector */}
      <div className="bg-white p-4 rounded-lg shadow">
        <div className="flex items-center space-x-4">
          <label className="text-sm font-medium text-gray-700">Time Range:</label>
          <select 
            value={timeRange} 
            onChange={(e) => setTimeRange(e.target.value)}
            className="border border-gray-300 rounded-md px-3 py-1 text-sm"
          >
            <option value="day">Last 24 hours</option>
            <option value="week">Last week</option>
            <option value="month">Last month</option>
          </select>
          {isLoading && (
            <div className="text-sm text-blue-600">
              🔄 Loading data...
            </div>
          )}
        </div>
      </div>

      {/* Sentiment Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="text-2xl font-bold text-green-600">
            {isLoading ? '...' : `${sentimentStats.positive}%`}
          </div>
          <div className="text-sm text-gray-500">Positive Sentiment</div>
          {!isLoading && sentimentData?.success && sentimentData.data && (
            <div className="text-xs text-gray-400 mt-1">
              {sentimentData.data.positive_count} posts
            </div>
          )}
        </div>
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="text-2xl font-bold text-red-600">
            {isLoading ? '...' : `${sentimentStats.negative}%`}
          </div>
          <div className="text-sm text-gray-500">Negative Sentiment</div>
          {!isLoading && sentimentData?.success && sentimentData.data && (
            <div className="text-xs text-gray-400 mt-1">
              {sentimentData.data.negative_count} posts
            </div>
          )}
        </div>
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="text-2xl font-bold text-yellow-600">
            {isLoading ? '...' : `${sentimentStats.neutral}%`}
          </div>
          <div className="text-sm text-gray-500">Neutral Sentiment</div>
          {!isLoading && sentimentData?.success && sentimentData.data && (
            <div className="text-xs text-gray-400 mt-1">
              {sentimentData.data.neutral_count} posts
            </div>
          )}
        </div>
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="text-2xl font-bold text-blue-600">
            {isLoading ? '...' : sentimentStats.total.toLocaleString()}
          </div>
          <div className="text-sm text-gray-500">Total Posts</div>
          {!isLoading && (
            <div className="text-xs text-gray-400 mt-1">
              in {timeRange === 'day' ? '24 hours' : timeRange === 'week' ? '7 days' : '30 days'}
            </div>
          )}
        </div>
      </div>

      {/* Charts Placeholder */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Sentiment Distribution</h3>
          <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center">
            {isLoading ? (
              <div className="text-center text-gray-500">
                <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-2"></div>
                <p>Loading sentiment data...</p>
              </div>
            ) : (
              <ResponsiveContainer width={200} height={200}>
                <PieChart>
                  <Pie
                    data={sentimentPieData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    paddingAngle={2}
                    startAngle={90}
                    endAngle={-270}
                  >
                    {sentimentPieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Top Impact Tweets</h3>
          <div className="h-64 overflow-y-auto">
            {isLoading ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center text-gray-500">
                  <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-2"></div>
                  <p>Loading top tweets...</p>
                </div>
              </div>
                         ) : parsedDashboardData?.topTweets && parsedDashboardData.topTweets.length > 0 ? (
               <div className="space-y-3">
                 {parsedDashboardData.topTweets.slice(0, 5).map((tweet: TweetData, index: number) => (
                  <div key={index} className="border-l-4 border-blue-500 pl-3 py-2">
                    <div className="text-sm font-medium text-gray-800">
                      @{tweet.author_username}
                    </div>
                    <div className="text-xs text-gray-600 mb-1">
                      Impact: {tweet.impact_score} | {tweet.sentiment}
                    </div>
                    <div className="text-sm text-gray-700">
                      {tweet.content.slice(0, 120)}...
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500">
                <div className="text-center">
                  <div className="text-4xl mb-2">🎯</div>
                  <p>No high-impact tweets found</p>
                  <p className="text-sm">Try a different time range</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Recent Posts */}
      <div className="bg-white p-6 rounded-lg shadow">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Recent Posts</h3>
          <select 
            value={sentimentFilter} 
            onChange={(e) => setSentimentFilter(e.target.value)}
            className="border border-gray-300 rounded-md px-3 py-1 text-sm"
          >
            <option value="">All sentiments</option>
            <option value="positive">Positive only</option>
            <option value="negative">Negative only</option>
            <option value="neutral">Neutral only</option>
          </select>
        </div>
        
        <div className="space-y-4 max-h-96 overflow-y-auto">
          {isLoading ? (
            <div className="text-center py-8">
              <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-2"></div>
              <p className="text-gray-500">Loading recent tweets...</p>
            </div>
          ) : parsedTweets.length > 0 ? (
            parsedTweets.map((tweet: TweetData, index: number) => (
              <div key={tweet.tweet_id || index} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-center space-x-3 mb-2">
                  <div className={`w-10 h-10 ${generateAvatarColor(tweet.author_username)} rounded-full flex items-center justify-center text-white font-bold text-sm`}>
                    {getAuthorInitials(tweet.author_username, tweet.author_name)}
                  </div>
                  <div>
                    <div className="font-medium">@{tweet.author_username}</div>
                    {tweet.author_name && tweet.author_name !== tweet.author_username && (
                      <div className="text-sm text-gray-600">{tweet.author_name}</div>
                    )}
                    <div className="text-sm text-gray-500">{formatTimeAgo(tweet.created_at)}</div>
                  </div>
                  <div className="ml-auto">
                    <span className={`px-2 py-1 rounded-full text-xs ${getSentimentColor(tweet.sentiment)}`}>
                      {tweet.sentiment.charAt(0).toUpperCase() + tweet.sentiment.slice(1)}
                    </span>
                  </div>
                </div>
                <p className="text-gray-700 mb-2">
                  {tweet.content}
                </p>
                <div className="flex items-center space-x-4 text-sm text-gray-500">
                  <span>❤️ {tweet.likes}</span>
                  <span>🔄 {tweet.retweets}</span>
                  <span>💬 {tweet.replies}</span>
                  <span>Impact: {tweet.impact_score.toFixed(1)}</span>
                  <span>Engagement: {tweet.engagement_score.toFixed(1)}%</span>
                </div>
              </div>
            ))
          ) : (
            <div className="text-center py-8 text-gray-500">
              <div className="text-4xl mb-2">📱</div>
              <p>No tweets found</p>
              <p className="text-sm">Try adjusting your filters or time range</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 