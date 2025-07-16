# Vaccine Hesitancy Voice Analyzer Agent

A comprehensive REST agent for analyzing audio recordings to identify vaccine hesitancy patterns, extract keywords with timestamps, and optionally detect COVID symptoms from voice characteristics.

## Features

- **Audio Transcription**: Converts MP3, WAV, M4A, and OGG files to text
- **Hesitancy Analysis**: Identifies 6 categories of vaccine hesitancy
- **Keyword Extraction**: Extracts important keywords with estimated timestamps
- **COVID Detection**: Optional voice-based COVID symptom detection (research only)
- **Text Analysis**: Analyze pre-transcribed text for hesitancy patterns
- **REST API**: Easy integration with web applications and frontends

## Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Install system dependencies (for audio processing):**

   **On macOS:**
   ```bash
   brew install ffmpeg portaudio
   ```

   **On Ubuntu/Debian:**
   ```bash
   sudo apt-get update
   sudo apt-get install ffmpeg portaudio19-dev python3-pyaudio
   ```

   **On Windows:**
   - Download FFmpeg from https://ffmpeg.org/
   - Add FFmpeg to your PATH
   - Install Visual C++ Build Tools if needed

3. **Set up environment variables:**
Create a `.env` file in the project root:
```
ASI1_API_KEY=your_asi1_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional backup
```

## Usage

### Starting the Agent

```bash
python vh_voice_analyzer_rest_agent.py
```

The agent will start on port 8003 and display available endpoints.

### REST API Endpoints

#### 1. Audio Analysis - `POST /analyze_audio`

Upload and analyze audio files for vaccine hesitancy.

**Request Body:**
```json
{
  "audio_data": "base64_encoded_audio_data",
  "filename": "recording.mp3",
  "content_type": "audio/mpeg",
  "analysis_options": {
    "hesitancy_analysis": true,
    "covid_detection": false,
    "keyword_extraction": true,
    "timestamp_analysis": true
  }
}
```

**Example using cURL:**
```bash
# First, encode your audio file to base64
base64 -i your_audio.mp3 -o audio_base64.txt

# Then send the request
curl -X POST http://localhost:8003/analyze_audio \
  -H "Content-Type: application/json" \
  -d '{
    "audio_data": "'$(cat audio_base64.txt)'",
    "filename": "your_audio.mp3",
    "content_type": "audio/mpeg",
    "analysis_options": {
      "hesitancy_analysis": true,
      "covid_detection": false,
      "keyword_extraction": true
    }
  }'
```

**Response:**
```json
{
  "request_id": "audio_1703123456",
  "success": true,
  "transcript": "I'm not sure about getting the vaccine because...",
  "transcript_confidence": 0.8,
  "hesitancy_analysis": {
    "overall_score": 0.456,
    "hesitancy_level": "moderate",
    "category_scores": {
      "safety_concerns": 3,
      "distrust_authorities": 1,
      "freedom_concerns": 2
    },
    "found_keywords": {
      "safety_concerns": ["side effects", "unsafe"],
      "freedom_concerns": ["personal choice"]
    },
    "analysis_summary": "Detected moderate vaccine hesitancy. Primary concerns: Safety Concerns, Freedom Concerns."
  },
  "keywords": [
    {
      "word": "vaccine",
      "timestamp": 5.2,
      "context": "getting the vaccine because I"
    },
    {
      "word": "side effects",
      "timestamp": 12.8,
      "context": "worried about side effects that"
    }
  ],
  "processing_time": 3.45,
  "audio_metadata": {
    "filename": "your_audio.mp3",
    "size_bytes": 245760,
    "format": "mp3"
  }
}
```

#### 2. Text Analysis - `POST /analyze_text`

Analyze pre-transcribed text for vaccine hesitancy patterns.

**Request:**
```json
{
  "text": "I don't trust the government and I'm worried about vaccine side effects...",
  "analysis_options": {
    "hesitancy_analysis": true,
    "keyword_extraction": true
  }
}
```

**Response:**
```json
{
  "request_id": "text_1703123456",
  "success": true,
  "hesitancy_analysis": {
    "overall_score": 0.625,
    "hesitancy_level": "moderate",
    "category_scores": {
      "safety_concerns": 2,
      "distrust_authorities": 1
    },
    "analysis_summary": "Detected moderate vaccine hesitancy. Primary concerns: Safety Concerns, Distrust Authorities."
  },
  "keywords": ["government", "worried", "vaccine", "side", "effects"],
  "processing_time": 0.12
}
```

#### 3. Health Check - `GET /health`

Check agent status and capabilities.

**Response:**
```json
{
  "status": "healthy",
  "agent_name": "Vaccine Hesitancy Voice Analyzer",
  "capabilities": [
    "Audio transcription (MP3, WAV, M4A, OGG)",
    "Vaccine hesitancy analysis",
    "Keyword extraction with timestamps",
    "Text-only analysis",
    "COVID voice detection"
  ],
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Postman Examples

#### Audio Analysis Request
1. Set method to `POST`
2. URL: `http://localhost:8003/analyze_audio`
3. Headers: `Content-Type: application/json`
4. Body (raw JSON):
```json
{
  "audio_data": "UklGRiQEAABXQVZFZm10IBAAAAABAAEAIlYAAESsAQACABAAZGF0YQAEAAA...",
  "filename": "test_recording.mp3",
  "content_type": "audio/mpeg",
  "analysis_options": {
    "hesitancy_analysis": true,
    "covid_detection": false,
    "keyword_extraction": true
  }
}
```

#### Text Analysis Request
1. Set method to `POST`
2. URL: `http://localhost:8003/analyze_text`
3. Headers: `Content-Type: application/json`
4. Body (raw JSON):
```json
{
  "text": "I'm really concerned about the vaccine side effects and I don't trust what the government is telling us about safety.",
  "analysis_options": {
    "hesitancy_analysis": true,
    "keyword_extraction": true
  }
}
```

## Analysis Categories

The agent identifies vaccine hesitancy in 6 main categories:

1. **Safety Concerns**: Side effects, adverse reactions, safety fears
2. **Distrust of Authorities**: Government control, big pharma concerns
3. **Religious Objections**: Faith-based exemptions, religious freedom
4. **Natural Immunity**: Preference for natural immunity, alternative medicine
5. **Freedom Concerns**: Personal choice, mandates, coercion
6. **Misinformation**: Conspiracy theories, false claims

## File Format Support

- **Primary**: MP3 (recommended for web uploads)
- **Supported**: WAV, M4A, OGG
- **Size Limit**: 0.5-3MB (1-3 minute recordings)
- **Quality**: 16kHz+ recommended for best transcription

## Transcription Services

The agent uses a fallback approach:
1. **Google Speech Recognition** (free, requires internet)
2. **Sphinx Offline** (backup, works offline)

## COVID Detection (Optional)

When enabled, the agent can analyze voice patterns for COVID-related symptoms using the HuggingFace model `greenarcade/cough-classification-model`. 

⚠️ **Important**: This is a research tool only and should not be used for medical diagnosis.

## Error Handling

The agent includes comprehensive error handling:
- Invalid audio formats
- Transcription failures
- Analysis errors
- Network issues

## Performance

- **Transcription**: 2-5 seconds for 1-minute audio
- **Analysis**: < 1 second for text processing
- **Total**: 3-6 seconds for complete audio analysis

## Integration Examples

### JavaScript/Frontend
```javascript
async function analyzeAudio(audioFile) {
  const formData = new FormData();
  
  // Convert file to base64
  const reader = new FileReader();
  reader.onload = async function(event) {
    const base64Audio = event.target.result.split(',')[1];
    
    const response = await fetch('http://localhost:8003/analyze_audio', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        audio_data: base64Audio,
        filename: audioFile.name,
        content_type: audioFile.type,
        analysis_options: {
          hesitancy_analysis: true,
          keyword_extraction: true
        }
      })
    });
    
    const result = await response.json();
    console.log('Analysis result:', result);
  };
  
  reader.readAsDataURL(audioFile);
}
```

### Python Client
```python
import base64
import requests

def analyze_audio_file(file_path):
    with open(file_path, 'rb') as f:
        audio_data = base64.b64encode(f.read()).decode('utf-8')
    
    response = requests.post('http://localhost:8003/analyze_audio', json={
        'audio_data': audio_data,
        'filename': file_path.split('/')[-1],
        'content_type': 'audio/mpeg',
        'analysis_options': {
            'hesitancy_analysis': True,
            'keyword_extraction': True
        }
    })
    
    return response.json()

# Usage
result = analyze_audio_file('recording.mp3')
print(f"Hesitancy level: {result['hesitancy_analysis']['hesitancy_level']}")
```

## Troubleshooting

### Common Issues

1. **"Could not understand audio"**
   - Check audio quality and volume
   - Ensure clear speech in recording
   - Try a different audio format

2. **"Invalid base64 audio data"**
   - Verify base64 encoding is correct
   - Check for data corruption during upload

3. **Transcription errors**
   - Ensure FFmpeg is installed
   - Check internet connection (for Google Speech)
   - Try shorter audio clips

4. **Import errors**
   - Install all requirements: `pip install -r requirements.txt`
   - Install system dependencies (FFmpeg, PortAudio)

### Debug Mode

Set environment variable for detailed logging:
```bash
export UAGENTS_LOG_LEVEL=DEBUG
python vh_voice_analyzer_rest_agent.py
```

## License

This project is for research and educational purposes only. The COVID detection feature is not intended for medical diagnosis. 