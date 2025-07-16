'use client';

import { useState, useRef, useEffect } from 'react';
import { voiceAnalyzerApi } from '@/lib/api/voice-analyzer';
import { AudioAnalysisResponse, TextAnalysisResponse } from '@/types';

const MAX_FILE_SIZE_MB = 10;
const ACCEPTED_FORMATS = ['audio/mp3', 'audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/m4a', 'audio/x-m4a'];
const LOCALSTORAGE_KEY = 'vh_voice_recent_analyses';

function getFileExtension(filename: string) {
  return filename.split('.').pop()?.toLowerCase() || '';
}

function isAcceptedFileType(file: File) {
  return ACCEPTED_FORMATS.includes(file.type) || ['mp3', 'wav', 'm4a'].includes(getFileExtension(file.name));
}

function formatDateAgo(dateString: string) {
  const date = new Date(dateString);
  const now = new Date();
  const diff = Math.floor((now.getTime() - date.getTime()) / 1000);
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

export function VoiceAnalysisPage() {
  const [analysisMode, setAnalysisMode] = useState<'audio' | 'text'>('audio');
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [transcript, setTranscript] = useState('');
  const [options, setOptions] = useState({
    hesitancy_analysis: true,
    keyword_extraction: true,
    sentiment_analysis: true,
  });
  const [result, setResult] = useState<AudioAnalysisResponse | TextAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recent, setRecent] = useState<any[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load recent analyses from localStorage
  useEffect(() => {
    const stored = localStorage.getItem(LOCALSTORAGE_KEY);
    if (stored) {
      setRecent(JSON.parse(stored));
    }
  }, []);

  // Save recent analyses to localStorage
  useEffect(() => {
    localStorage.setItem(LOCALSTORAGE_KEY, JSON.stringify(recent.slice(0, 5)));
  }, [recent]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    setResult(null);
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (!isAcceptedFileType(file)) {
        setError('Unsupported file type. Please upload MP3, WAV, or M4A.');
        setAudioFile(null);
        return;
      }
      if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
        setError('File is too large. Max 10MB allowed.');
        setAudioFile(null);
        return;
      }
      setAudioFile(file);
    }
  };

  const handleSubmit = async () => {
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      if (analysisMode === 'audio') {
        if (!audioFile) {
          setError('Please select an audio file.');
          setLoading(false);
          return;
        }
        // Read file as base64
        const reader = new FileReader();
        reader.onload = async () => {
          const base64 = (reader.result as string).split(',')[1];
          const req = {
            audio_data: base64,
            filename: audioFile.name,
            content_type: audioFile.type,
            analysis_options: {
              hesitancy_analysis: options.hesitancy_analysis,
              keyword_extraction: options.keyword_extraction,
            },
          };
          const resp = await voiceAnalyzerApi.analyzeAudio(req);
          if (resp.success && resp.data) {
            const data = resp.data;
            setResult(data);
            setRecent((prev) => [{
              ...data,
              mode: 'audio',
              filename: audioFile.name,
              timestamp: data.timestamp || new Date().toISOString(),
            }, ...prev].slice(0, 5));
          } else {
            setError(resp.error || 'Analysis failed.');
          }
          setLoading(false);
        };
        reader.onerror = () => {
          setError('Failed to read audio file.');
          setLoading(false);
        };
        reader.readAsDataURL(audioFile);
      } else {
        if (!transcript.trim()) {
          setError('Please enter transcript text.');
          setLoading(false);
          return;
        }
        const req = {
          text: transcript,
          analysis_options: {
            hesitancy_analysis: options.hesitancy_analysis,
            keyword_extraction: options.keyword_extraction,
            sentiment_analysis: options.sentiment_analysis,
          },
        };
        const resp = await voiceAnalyzerApi.analyzeText(req);
        if (resp.success && resp.data) {
          const data = resp.data;
          setResult(data);
          setRecent((prev) => [{
            ...data,
            mode: 'text',
            transcript,
            timestamp: data.timestamp || new Date().toISOString(),
          }, ...prev].slice(0, 5));
        } else {
          setError(resp.error || 'Analysis failed.');
        }
        setLoading(false);
      }
    } catch (e: any) {
      setError(e.message || 'Unexpected error.');
      setLoading(false);
    }
  };

  const handleClear = () => {
    setAudioFile(null);
    setTranscript('');
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleDeleteRecent = (idx: number) => {
    setRecent((prev) => prev.filter((_, i) => i !== idx));
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold text-gray-900">Voice Analysis</h2>
        <div className="text-sm text-gray-500">
          Audio transcription and vaccine hesitancy analysis
        </div>
      </div>

      {/* Mode Selection */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Analysis Mode</h3>
        <div className="flex space-x-4">
          <button
            onClick={() => { setAnalysisMode('audio'); handleClear(); }}
            className={`px-6 py-3 rounded-lg font-medium transition-colors ${
              analysisMode === 'audio'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            🎤 Audio Analysis
          </button>
          <button
            onClick={() => { setAnalysisMode('text'); handleClear(); }}
            className={`px-6 py-3 rounded-lg font-medium transition-colors ${
              analysisMode === 'text'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            📝 Text Analysis
          </button>
        </div>
      </div>

      {/* Analysis Input */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Panel */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">
            {analysisMode === 'audio' ? 'Audio Input' : 'Text Input'}
          </h3>
          {analysisMode === 'audio' ? (
            <div className="space-y-4">
              <div className="border border-gray-300 rounded-lg p-4">
                <h4 className="font-medium mb-2">Upload Audio File</h4>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".mp3,.wav,.m4a,audio/mp3,audio/mpeg,audio/wav,audio/x-wav,audio/m4a,audio/x-m4a"
                  className="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                  onChange={handleFileChange}
                />
                <p className="text-xs text-gray-500 mt-1">
                  Supported formats: MP3, WAV, M4A. Max size: 10MB.
                </p>
                {audioFile && (
                  <div className="mt-2 text-xs text-gray-700">
                    Selected: {audioFile.name} ({(audioFile.size / 1024 / 1024).toFixed(2)} MB)
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <textarea
                className="w-full h-48 border border-gray-300 rounded-lg p-4 text-sm"
                placeholder="Enter text for vaccine hesitancy analysis..."
                value={transcript}
                onChange={e => setTranscript(e.target.value)}
              />
            </div>
          )}
          <button
            className="w-full py-3 mt-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium"
            onClick={handleSubmit}
            disabled={loading}
          >
            {loading ? 'Analyzing...' : 'Submit'}
          </button>
          <button
            className="w-full py-2 mt-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 font-medium"
            onClick={handleClear}
            disabled={loading}
          >
            Clear
          </button>
          {error && (
            <div className="mt-2 text-red-600 text-sm">{error}</div>
          )}
        </div>

        {/* Results Panel */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Analysis Results</h3>
          {loading ? (
            <div className="text-center text-blue-600">Analyzing...</div>
          ) : result ? (
            <div className="space-y-4">
              {'transcript' in result && (
                <div className="border border-gray-200 rounded-lg p-4">
                  <h4 className="font-medium mb-2">Transcription</h4>
                  <p className="text-sm text-gray-600 bg-gray-50 p-3 rounded">
                    {result.transcript || '[No transcript]'}
                  </p>
                  <div className="text-xs text-gray-500 mt-2">
                    Confidence: {result.transcript_confidence ? `${Math.round(result.transcript_confidence * 100)}%` : 'N/A'}
                  </div>
                </div>
              )}
              {'hesitancy_analysis' in result && (
                <div className="border border-gray-200 rounded-lg p-4">
                  <h4 className="font-medium mb-2">Hesitancy Analysis</h4>
                  {/* Summary - bold and large */}
                  <div className="mb-4">
                    <span className="block text-lg font-bold text-gray-900">
                      {result.hesitancy_analysis.analysis_summary ?? 'N/A'}
                    </span>
                  </div>
                  {/* Score and Level - right aligned */}
                  <div className="flex justify-end space-x-8 mb-2">
                    <div className="text-right">
                      <div className="text-xs text-gray-500">Hesitancy Score</div>
                      <div className="text-2xl font-bold text-orange-600">{result.hesitancy_analysis.hesitancy_score ?? 'N/A'}</div>
                    </div>
                    <div className="text-right">
                      <div className="text-xs text-gray-500">Level</div>
                      <div className="text-xl font-bold text-green-600">{result.hesitancy_analysis.hesitancy_level ?? 'N/A'}</div>
                    </div>
                  </div>
                  {/* Categories */}
                  {Array.isArray(result.hesitancy_analysis.identified_categories) && result.hesitancy_analysis.identified_categories.length > 0 && (
                    <div className="mt-4">
                      <div className="font-medium text-sm mb-2">Identified Categories:</div>
                      <div className="space-y-2">
                        {result.hesitancy_analysis.identified_categories.map((cat: any, idx: number) => (
                          <div key={idx} className="flex flex-col md:flex-row md:items-center md:space-x-2 bg-gray-50 rounded p-2">
                            <div className="font-semibold text-blue-700 mr-2">{cat.category}</div>
                            <div className="text-xs text-gray-500 mr-2">({cat.strength})</div>
                            {cat.sample_sentences && cat.sample_sentences.length > 0 && (
                              <div className="text-xs text-gray-700 mt-1 md:mt-0">{cat.sample_sentences.join('; ')}</div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
              {'keywords' in result && Array.isArray(result.keywords) && (
                <div className="border border-gray-200 rounded-lg p-4">
                  <h4 className="font-medium mb-2">Extracted Keywords</h4>
                  <div className="flex flex-wrap gap-2">
                    {result.keywords.map((kw: string, idx: number) => (
                      <span key={idx} className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-semibold shadow-sm border border-blue-200">{kw}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-gray-400 text-center">No analysis yet.</div>
          )}
        </div>
      </div>

      {/* Analysis Options */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Analysis Options</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <label className="flex items-center space-x-2">
            <input type="checkbox" checked={options.hesitancy_analysis} onChange={e => setOptions(o => ({ ...o, hesitancy_analysis: e.target.checked }))} className="rounded" />
            <span className="text-sm">Hesitancy Analysis</span>
          </label>
          <label className="flex items-center space-x-2">
            <input type="checkbox" checked={options.keyword_extraction} onChange={e => setOptions(o => ({ ...o, keyword_extraction: e.target.checked }))} className="rounded" />
            <span className="text-sm">Keyword Extraction</span>
          </label>
          <label className="flex items-center space-x-2">
            <input type="checkbox" checked={options.sentiment_analysis} onChange={e => setOptions(o => ({ ...o, sentiment_analysis: e.target.checked }))} className="rounded" />
            <span className="text-sm">Sentiment Analysis</span>
          </label>
        </div>
      </div>

      {/* Recent Analyses */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Recent Analyses</h3>
        <div className="space-y-3">
          {recent.length === 0 && <div className="text-gray-400 text-center">No recent analyses.</div>}
          {recent.map((item, idx) => (
            <div key={item.request_id || idx} className="flex items-center justify-between p-3 border border-gray-200 rounded-lg">
              <div className="flex items-center space-x-3">
                <div className="text-2xl">{item.mode === 'audio' ? '🎤' : '📝'}</div>
                <div>
                  <div className="font-medium">{item.mode === 'audio' ? (item.filename || 'Audio Analysis') : 'Text Analysis'}</div>
                  <div className="text-sm text-gray-500">{formatDateAgo(item.timestamp)} • Hesitancy Score: {item.hesitancy_analysis?.hesitancy_score ?? 'N/A'}</div>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <button className="text-blue-600 hover:text-blue-800 text-sm" onClick={() => setResult(item)}>
                  View Details
                </button>
                <button className="text-red-500 hover:text-red-700 text-sm" onClick={() => handleDeleteRecent(idx)}>
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
} 