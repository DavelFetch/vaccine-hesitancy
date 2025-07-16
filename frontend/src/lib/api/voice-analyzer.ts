import { BaseApiClient } from './base';
import { config } from '@/lib/config';
import { ApiResponse, AudioUploadRequest, AudioAnalysisResponse, TextAnalysisRequest, TextAnalysisResponse } from '@/types';

export class VoiceAnalyzerApi extends BaseApiClient {
  constructor() {
    super(config.agents.voiceAnalyzer);
  }

  async analyzeAudio(request: AudioUploadRequest): Promise<ApiResponse<AudioAnalysisResponse>> {
    return this.post<AudioAnalysisResponse>('/analyze_audio', request);
  }

  async analyzeText(request: TextAnalysisRequest): Promise<ApiResponse<TextAnalysisResponse>> {
    return this.post<TextAnalysisResponse>('/analyze_text', request);
  }
}

export const voiceAnalyzerApi = new VoiceAnalyzerApi(); 