export const config = {
  apiBaseUrl: process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost',
  mapboxToken: process.env.NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN || '',
  agents: {
    hesitancyInsights: process.env.NEXT_PUBLIC_HESITANCY_AGENT_URL || 'http://localhost:8005',
    xAnalysis: process.env.NEXT_PUBLIC_X_ANALYSIS_AGENT_URL || 'http://localhost:8001',
    vaccineResource: process.env.NEXT_PUBLIC_VACCINE_RESOURCE_AGENT_URL || 'http://localhost:8002',
    voiceAnalyzer: process.env.NEXT_PUBLIC_VOICE_ANALYZER_AGENT_URL || 'http://localhost:8004',
  },
} as const;

export type AgentType = keyof typeof config.agents; 