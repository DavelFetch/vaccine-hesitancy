from dotenv import load_dotenv
from uagents import Agent, Context, Model
from pydantic import Field
from datetime import datetime, timezone, timedelta
import json
import base64
import asyncio
from typing import Dict, List, Optional, Any, Union
import os
import re
import requests
import tempfile
import wave
import librosa
import numpy as np
import pandas as pd
from collections import Counter
import speech_recognition as sr
from pydub import AudioSegment
import pickle
from huggingface_hub import hf_hub_download

# Load environment variables
load_dotenv()

# Validate required environment variables
required_env_vars = {
    "ASI1_API_KEY": os.getenv("ASI1_API_KEY"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")  # Backup for transcription
}

# Check if any required variables are missing
missing_vars = [var for var, value in required_env_vars.items() if not value]
if missing_vars:
    print(f"Warning: Missing environment variables: {', '.join(missing_vars)}")

# ASI1 configuration
ASI1_URL = "https://api.asi1.ai/v1/chat/completions"
ASI1_HEADERS = {
    "Authorization": f"Bearer {required_env_vars.get('ASI1_API_KEY', '')}",
    "Content-Type": "application/json"
}

# Models for REST endpoints
class AudioUploadRequest(Model):
    """Request model for audio file upload"""
    audio_data: str = Field(..., description="Base64 encoded audio file data")
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(default="audio/mpeg", description="MIME type of audio file")
    analysis_options: Dict[str, bool] = Field(
        default_factory=lambda: {
            "hesitancy_analysis": True,
            "keyword_extraction": True
        },
        description="Analysis options to enable"
    )

class AudioAnalysisResponse(Model):
    """Response model for audio analysis results"""
    request_id: str = Field(..., description="Unique request identifier")
    success: bool = Field(..., description="Analysis success status")
    
    # Transcription results
    transcript: str = Field(default="", description="Full transcribed text")
    transcript_confidence: float = Field(default=0.0, description="Transcription confidence score")
    
    # Hesitancy analysis
    hesitancy_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Vaccine hesitancy analysis results"
    )
    
    # Keywords
    keywords: List[str] = Field(
        default_factory=list,
        description="Extracted keywords"
    )
    
    # Metadata
    processing_time: float = Field(..., description="Total processing time in seconds")
    audio_metadata: Dict[str, Any] = Field(default_factory=dict, description="Audio file metadata")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class HealthResponse(Model):
    """Health check response"""
    status: str = Field(..., description="Health status")
    agent_name: str = Field(..., description="Agent name")
    capabilities: List[str] = Field(..., description="Available capabilities")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class TextAnalysisRequest(Model):
    """Request model for analyzing pre-transcribed text"""
    text: str = Field(..., description="Text to analyze for vaccine hesitancy")
    analysis_options: Dict[str, bool] = Field(
        default_factory=lambda: {
            "hesitancy_analysis": True,
            "keyword_extraction": True,
            "sentiment_analysis": True
        },
        description="Analysis options to enable"
    )

class TextAnalysisResponse(Model):
    """Response model for text analysis"""
    request_id: str = Field(..., description="Unique request identifier")
    success: bool = Field(..., description="Analysis success status")
    hesitancy_analysis: Dict[str, Any] = Field(default_factory=dict, description="Analysis results")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class CovidAnalysisRequest(Model):
    """Request model for COVID voice analysis"""
    audio_data: str = Field(..., description="Base64 encoded audio file data")
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(default="audio/mpeg", description="MIME type of audio file")

class CovidAnalysisResponse(Model):
    """Response model for COVID voice analysis"""
    request_id: str = Field(..., description="Unique request identifier")
    success: bool = Field(..., description="Analysis success status")
    covid_analysis: Optional[Dict[str, Any]] = Field(
        default=None,
        description="COVID detection results from voice"
    )
    processing_time: float = Field(..., description="Processing time in seconds")
    audio_metadata: Dict[str, Any] = Field(default_factory=dict, description="Audio file metadata")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class VoiceAnalyzer:
    """Main voice analysis class with ASI1-powered hesitancy detection"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.covid_model = None
        # Initialize hesitancy keywords for compatibility
        self.hesitancy_keywords = {
            "safety_concerns": ["side", "effect", "adverse", "reaction", "risk", "danger", "harm", "safe", "safety"],
            "distrust_authorities": ["government", "pharma", "conspiracy", "lie", "trust", "corrupt", "control"],
            "religious_objections": ["faith", "belief", "religious", "conscience", "moral", "sin", "god"],
            "natural_immunity": ["natural", "immunity", "immune", "body", "defense", "antibodies"],
            "freedom_concerns": ["choice", "freedom", "liberty", "force", "mandate", "coercion", "rights"],
            "misinformation": ["chip", "tracking", "dna", "genetic", "experiment", "population", "control"]
        }
        self.load_covid_model()

    def load_covid_model(self):
        """Load COVID detection model from HuggingFace"""
        try:
            model_path = hf_hub_download(
                repo_id="greenarcade/cough-classification-model",
                filename="cough_classification_model.pkl"
            )
            
            with open(model_path, 'rb') as f:
                components = pickle.load(f)
                
            self.covid_model = {
                'model': components['model'],
                'scaler': components['scaler'],
                'label_encoder': components['label_encoder'],
                'feature_names': components['feature_names']
            }
            print("✅ COVID detection model loaded successfully")
        except Exception as e:
            print(f"⚠️ COVID model loading failed: {e}")
            self.covid_model = None

    async def analyze_hesitancy_with_asi1(self, text: str) -> Dict[str, Any]:
        """Analyze vaccine hesitancy using ASI1 for intelligent understanding"""
        try:
            # Focused, practical prompt that won't cause hallucinations
            system_prompt = """You are a vaccine hesitancy analyst. Analyze text for vaccine hesitancy patterns and respond with valid JSON only.

CATEGORIES:
- safety_concerns: worries about side effects, safety, adverse reactions
- distrust_authorities: skepticism of government, medical establishment, pharmaceutical companies
- religious_objections: faith-based or conscience objections
- natural_immunity: preference for natural immune response
- freedom_concerns: personal choice, mandate resistance, coercion issues
- misinformation: conspiracy theories, false beliefs

SCORING: 0-100% hesitancy level
- 0-25%: minimal (mostly supportive, minor concerns)
- 26-50%: moderate (mixed feelings, some concerns)  
- 51-75%: high (significant concerns, reluctance)
- 76-100%: extreme (strong opposition, conspiracy beliefs)

RESPONSE FORMAT (JSON only):
{
  "hesitancy_score": 65,
  "hesitancy_level": "high",
  "analysis_summary": "Brief 1-2 sentence summary of findings",
  "identified_categories": [
    {
      "category": "freedom_concerns",
      "strength": "high", 
      "sample_sentences": ["exact quote 1", "exact quote 2"]
    }
  ]
}

Extract actual quotes from the text. Keep sample_sentences under 100 characters each."""

            payload = {
                "model": "asi1-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this text for vaccine hesitancy:\n\n{text}"}
                ],
                "temperature": 0.2,
                "max_tokens": 500
            }
            
            response = requests.post(ASI1_URL, headers=ASI1_HEADERS, json=payload)
            
            if response.status_code != 200:
                return self._fallback_analysis(f"ASI1 API error: {response.status_code}")
            
            result_text = response.json()["choices"][0]["message"]["content"].strip()
            
            # Clean JSON response (remove markdown if present)
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].strip()
            
            # Parse JSON response
            try:
                analysis_result = json.loads(result_text)
                
                # Validate required fields
                required_fields = ["hesitancy_score", "hesitancy_level", "analysis_summary", "identified_categories"]
                if not all(field in analysis_result for field in required_fields):
                    print(f"❌ Missing fields in ASI1 response: {analysis_result}")
                    return self._fallback_analysis("Missing required fields in ASI1 response")
                
                print(f"✅ ASI1 analysis result: {analysis_result}")
                return analysis_result
                
            except json.JSONDecodeError as e:
                return self._fallback_analysis(f"Invalid JSON from ASI1: {str(e)}")
                
        except Exception as e:
            return self._fallback_analysis(f"ASI1 analysis failed: {str(e)}")

    def _fallback_analysis(self, error_msg: str) -> Dict[str, Any]:
        """Fallback analysis when ASI1 fails"""
        return {
            "hesitancy_score": 0,
            "hesitancy_level": "unknown",
            "analysis_summary": f"Analysis failed: {error_msg}",
            "identified_categories": [],
            "error": error_msg
        }

    def extract_simple_keywords(self, text: str) -> List[str]:
        """Simple keyword extraction without timestamps for text analysis"""
        try:
            words = text.lower().split()
            important_words = []
            
            # Focus on vaccine-related and significant terms
            vaccine_terms = ['vaccine', 'vaccination', 'shot', 'dose', 'covid', 'pfizer', 'moderna', 'hesitancy', 'safety', 'side', 'effects']
            
            for word in words:
                clean_word = re.sub(r'[^\w\s]', '', word)
                if (len(clean_word) > 4 or 
                    any(term in clean_word for term in vaccine_terms)):
                    important_words.append(clean_word)
            
            # Return unique keywords, limited
            return list(set(important_words))[:20]
            
        except Exception as e:
            return [f"keyword_extraction_error: {str(e)}"]

    def extract_audio_features(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Extract audio features for COVID detection - enhanced version matching model requirements"""
        if not self.covid_model:
            return None
            
        try:
            # Save temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(audio_data)
            
            try:
                # Load audio with consistent sample rate
                y, sr = librosa.load(temp_path, sr=22050)  # Standardize sample rate
                
                # Extract comprehensive features matching the model requirements
                features = {}
                
                # Temporal features
                features['rms_mean'] = np.mean(librosa.feature.rms(y=y))
                features['zcr_mean'] = np.mean(librosa.feature.zero_crossing_rate(y=y))
                
                # Spectral features
                features['spectral_centroid_mean'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                features['spectral_bandwidth_mean'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
                features['spectral_rolloff_mean'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
                
                # Add spectral contrast (was missing)
                spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                features['spectral_contrast_mean'] = np.mean(spectral_contrast)
                
                # Chroma features
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                features['chroma_mean'] = np.mean(chroma)
                
                # MFCCs (13 coefficients with means and standard deviations)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                for i, mfcc in enumerate(mfccs):
                    features[f'mfcc{i+1}_mean'] = np.mean(mfcc)
                    features[f'mfcc{i+1}_std'] = np.std(mfcc)
                
                # Additional features that might be in the model
                # Tonnetz (harmonic features)
                tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
                features['tonnetz_mean'] = np.mean(tonnetz)
                
                # Tempo and beat features
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                features['tempo'] = tempo
                
                # Zero crossing rate std
                zcr = librosa.feature.zero_crossing_rate(y=y)
                features['zcr_std'] = np.std(zcr)
                
                # RMS std
                rms = librosa.feature.rms(y=y)
                features['rms_std'] = np.std(rms)
                
                # Create DataFrame and filter to model features
                features_df = pd.DataFrame([features])
                
                # Only use features that the model expects
                if 'feature_names' in self.covid_model:
                    available_features = [f for f in self.covid_model['feature_names'] if f in features_df.columns]
                    features_df = features_df[available_features]
                    
                    # Fill missing features with 0 if any are missing
                    for feature_name in self.covid_model['feature_names']:
                        if feature_name not in features_df.columns:
                            features_df[feature_name] = 0.0
                    
                    # Ensure correct order
                    features_df = features_df[self.covid_model['feature_names']]
                
                print(f"✅ Extracted {len(features_df.columns)} features for COVID detection")
                return features_df
                
            finally:
                os.unlink(temp_path)
                
        except Exception as e:
            print(f"❌ Error extracting audio features: {e}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            return None

    def detect_covid_from_voice(self, audio_data: bytes) -> Optional[Dict[str, Any]]:
        """Detect COVID symptoms from voice patterns"""
        if not self.covid_model:
            print("❌ COVID model not loaded")
            return None
            
        try:
            print("🔍 Extracting audio features...")
            features = self.extract_audio_features(audio_data)
            if features is None:
                print("❌ Feature extraction failed")
                return None
            
            print(f"✅ Features extracted: {features.shape}")
            
            # Scale features
            print("🔧 Scaling features...")
            features_scaled = self.covid_model['scaler'].transform(features)
            print(f"✅ Features scaled: {features_scaled.shape}")
            
            # Get predictions
            print("🤖 Making predictions...")
            prediction_idx = self.covid_model['model'].predict(features_scaled)[0]
            probabilities = self.covid_model['model'].predict_proba(features_scaled)[0]
            
            # Convert to class labels
            predicted_class = self.covid_model['label_encoder'].inverse_transform([prediction_idx])[0]
            class_probabilities = {
                self.covid_model['label_encoder'].inverse_transform([i])[0]: float(prob)
                for i, prob in enumerate(probabilities)
            }
            
            print(f"✅ Prediction complete: {predicted_class} (confidence: {max(probabilities):.3f})")
            
            return {
                "predicted_class": predicted_class,
                "probabilities": class_probabilities,
                "confidence": float(max(probabilities)),
                "note": "This is a research tool, not for medical diagnosis"
            }
            
        except Exception as e:
            print(f"❌ Error in COVID detection: {e}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            return None

    def convert_audio_to_wav(self, audio_data: bytes, original_format: str) -> bytes:
        """Convert audio to WAV format for processing"""
        try:
            # Create temporary input file
            with tempfile.NamedTemporaryFile(suffix=f'.{original_format}', delete=False) as temp_input:
                temp_input.write(audio_data)
                temp_input_path = temp_input.name
            
            try:
                # Convert to WAV using pydub
                if original_format.lower() in ['mp3', 'mpeg']:
                    audio = AudioSegment.from_mp3(temp_input_path)
                elif original_format.lower() in ['m4a', 'mp4']:
                    audio = AudioSegment.from_file(temp_input_path, format="mp4")
                elif original_format.lower() == 'ogg':
                    audio = AudioSegment.from_ogg(temp_input_path)
                else:
                    # Assume it's already WAV or try generic
                    audio = AudioSegment.from_file(temp_input_path)
                
                # Export as WAV
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
                    temp_output_path = temp_output.name
                    audio.export(temp_output_path, format="wav")
                
                # Read WAV data
                with open(temp_output_path, 'rb') as f:
                    wav_data = f.read()
                
                os.unlink(temp_output_path)
                return wav_data
                
            finally:
                os.unlink(temp_input_path)
                
        except Exception as e:
            print(f"Error converting audio: {e}")
            return audio_data  # Return original if conversion fails

    def transcribe_audio(self, audio_data: bytes, original_format: str = 'mp3') -> Dict[str, Any]:
        """Transcribe audio to text using speech recognition"""
        try:
            # Convert to WAV if needed
            if original_format.lower() != 'wav':
                audio_data = self.convert_audio_to_wav(audio_data, original_format)
            
            # Save temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(audio_data)
            
            try:
                # Transcribe using speech_recognition
                with sr.AudioFile(temp_path) as source:
                    # Adjust for ambient noise
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    # Record the audio
                    audio = self.recognizer.record(source)
                
                # Try Google Speech Recognition (free tier)
                try:
                    transcript = self.recognizer.recognize_google(audio)
                    confidence = 0.8  # Google doesn't provide confidence scores
                    
                    return {
                        "transcript": transcript,
                        "confidence": confidence,
                        "method": "google_speech"
                    }
                except sr.UnknownValueError:
                    return {
                        "transcript": "",
                        "confidence": 0.0,
                        "error": "Could not understand audio"
                    }
                except sr.RequestError as e:
                    # Fallback to offline recognition
                    try:
                        transcript = self.recognizer.recognize_sphinx(audio)
                        return {
                            "transcript": transcript,
                            "confidence": 0.6,
                            "method": "sphinx_offline"
                        }
                    except:
                        return {
                            "transcript": "",
                            "confidence": 0.0,
                            "error": f"Speech recognition failed: {str(e)}"
                        }
                        
            finally:
                os.unlink(temp_path)
                
        except Exception as e:
            return {
                "transcript": "",
                "confidence": 0.0,
                "error": f"Transcription error: {str(e)}"
            }

    def extract_keywords_with_timestamps(self, transcript: str, audio_duration: float = None) -> List[Dict[str, Any]]:
        """Extract important keywords and estimate timestamps"""
        try:
            # Simple keyword extraction
            words = transcript.lower().split()
            word_count = Counter(words)
            
            # Filter out common words and extract vaccine-related terms
            vaccine_terms = []
            hesitancy_terms = []
            
            for word in words:
                # Remove punctuation
                clean_word = re.sub(r'[^\w\s]', '', word)
                if len(clean_word) < 3:
                    continue
                    
                # Check if it's a vaccine-related term
                if any(term in clean_word for term in ['vaccin', 'shot', 'dose', 'immun', 'covid', 'pfizer', 'moderna']):
                    vaccine_terms.append(clean_word)
                
                # Check hesitancy keywords
                for category, keywords in self.hesitancy_keywords.items():
                    if any(keyword in clean_word for keyword in keywords):
                        hesitancy_terms.append({
                            'word': clean_word,
                            'category': category
                        })
            
            # Estimate timestamps (simplified)
            keywords_with_timestamps = []
            if audio_duration:
                words_per_second = len(words) / audio_duration
                
                for i, word in enumerate(words):
                    clean_word = re.sub(r'[^\w\s]', '', word.lower())
                    
                    # Only include significant words
                    if (len(clean_word) > 4 or 
                        any(term in clean_word for term in ['vaccin', 'covid', 'shot']) or
                        any(any(kw in clean_word for kw in keywords) for keywords in self.hesitancy_keywords.values())):
                        
                        timestamp = i / words_per_second
                        keywords_with_timestamps.append({
                            'word': clean_word,
                            'timestamp': round(timestamp, 1),
                            'context': ' '.join(words[max(0, i-2):i+3])
                        })
            
            return keywords_with_timestamps[:20]  # Limit to top 20
            
        except Exception as e:
            return [{"error": f"Keyword extraction failed: {str(e)}"}]

# Initialize agent
agent = Agent(
    name="vaccine_hesitancy_voice_analyzer",
    port=8004,
    seed="vh_voice_analyzer_rest_agent_seed",
    mailbox=False  # REST-only agent
)

# Initialize analyzer
analyzer = VoiceAnalyzer()

@agent.on_event("startup")
async def startup_function(ctx: Context):
    ctx.logger.info("🎤 Starting Vaccine Hesitancy Voice Analyzer Agent")
    ctx.logger.info(f"📍 Agent address: {agent.address}")
    ctx.logger.info("🔧 Available endpoints:")
    ctx.logger.info("   POST /analyze_audio - Upload and analyze audio files for hesitancy")
    ctx.logger.info("   POST /analyze_text  - Analyze pre-transcribed text")
    ctx.logger.info("   POST /analyze_covid - COVID voice detection from audio")
    ctx.logger.info("   GET  /health        - Health check")
    ctx.logger.info("✅ Agent ready for voice analysis!")

@agent.on_rest_post("/analyze_audio", AudioUploadRequest, AudioAnalysisResponse)
async def handle_audio_analysis(ctx: Context, request: AudioUploadRequest) -> AudioAnalysisResponse:
    """Handle audio file upload and analysis"""
    start_time = datetime.now()
    request_id = f"audio_{int(start_time.timestamp())}"
    
    try:
        ctx.logger.info(f"📤 Received audio analysis request: {request.filename}")
        
        # Decode base64 audio data
        try:
            audio_data = base64.b64decode(request.audio_data)
        except Exception as e:
            return AudioAnalysisResponse(
                request_id=request_id,
                success=False,
                transcript="",
                hesitancy_analysis={"error": f"Invalid base64 audio data: {str(e)}"},
                processing_time=0.0
            )
        
        # Extract file format
        file_format = request.filename.split('.')[-1].lower() if '.' in request.filename else 'mp3'
        
        # Get audio metadata
        audio_metadata = {
            "filename": request.filename,
            "size_bytes": len(audio_data),
            "format": file_format,
            "content_type": request.content_type
        }
        
        ctx.logger.info(f"🎵 Audio metadata: {audio_metadata}")
        
        # Step 1: Transcribe audio
        ctx.logger.info("🗣️ Starting transcription...")
        transcription_result = analyzer.transcribe_audio(audio_data, file_format)
        
        # Log the full transcription result
        ctx.logger.info(f"🔍 Transcription result type: {type(transcription_result)}")
        ctx.logger.info(f"🔍 Transcription result keys: {list(transcription_result.keys()) if isinstance(transcription_result, dict) else 'Not a dict'}")
        ctx.logger.info(f"🔍 Transcription result: {transcription_result}")
        
        transcript = transcription_result.get("transcript", "")
        transcript_confidence = transcription_result.get("confidence", 0.0)
        
        ctx.logger.info(f"🔍 Extracted transcript length: {len(transcript)}")
        ctx.logger.info(f"🔍 Extracted confidence: {transcript_confidence}")
        ctx.logger.info(f"🔍 Transcript confidence type: {type(transcript_confidence)}")
        
        if not transcript:
            error_msg = transcription_result.get("error", "Transcription failed")
            ctx.logger.error(f"❌ Transcription failed: {error_msg}")
            return AudioAnalysisResponse(
                request_id=request_id,
                success=False,
                transcript="",
                transcript_confidence=0.0,
                hesitancy_analysis={"error": error_msg},
                processing_time=0.0,
                audio_metadata=audio_metadata
            )
        
        ctx.logger.info(f"✅ Transcription completed: {len(transcript)} characters")
        ctx.logger.info(f"📝 Transcript preview: {transcript[:100]}...")
        
        # Get analysis options as dict
        analysis_opts = dict(request.analysis_options) if hasattr(request.analysis_options, 'items') else request.analysis_options
        
        # Step 2: Analyze hesitancy (if enabled)
        hesitancy_analysis = {}
        if analysis_opts.get("hesitancy_analysis", True):
            ctx.logger.info("🔍 Analyzing vaccine hesitancy patterns...")
            hesitancy_analysis = await analyzer.analyze_hesitancy_with_asi1(transcript)
        
        # Step 3: Extract keywords (if enabled)
        keywords = []
        if analysis_opts.get("keyword_extraction", True):
            ctx.logger.info("🔑 Extracting keywords...")
            keywords = analyzer.extract_simple_keywords(transcript)
            # Ensure keywords is a list of strings
            if keywords and isinstance(keywords[0], dict):
                keywords = [k.get('word', str(k)) for k in keywords]

        processing_time = (datetime.now() - start_time).total_seconds()
        
        ctx.logger.info(f"✅ Analysis completed in {processing_time:.2f} seconds")
        
        # Debug logging before creating response
        ctx.logger.info(f"🔍 Debug - About to create response:")
        ctx.logger.info(f"   hesitancy_analysis type: {type(hesitancy_analysis)}")
        ctx.logger.info(f"   keywords type: {type(keywords)}")
        ctx.logger.info(f"   processing_time type: {type(processing_time)}")
        
        try:
            response = AudioAnalysisResponse(
                request_id=request_id,
                success=True,
                transcript=transcript,
                transcript_confidence=transcript_confidence,
                hesitancy_analysis=hesitancy_analysis,
                keywords=keywords,
                processing_time=processing_time,
                audio_metadata=audio_metadata,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            ctx.logger.info(f"✅ Response created successfully")
            return response
        except Exception as e:
            ctx.logger.error(f"❌ Failed to create response: {str(e)}")
            ctx.logger.error(f"❌ Exception type: {type(e)}")
            # Return a minimal response that should work
            return AudioAnalysisResponse(
                request_id=request_id,
                success=False,
                transcript=transcript,
                transcript_confidence=transcript_confidence,
                hesitancy_analysis={"error": f"Response creation failed: {str(e)}"},
                keywords=[],
                processing_time=processing_time,
                audio_metadata=audio_metadata,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
    except Exception as e:
        ctx.logger.error(f"❌ Audio analysis failed: {str(e)}")
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AudioAnalysisResponse(
            request_id=request_id,
            success=False,
            transcript="",
            hesitancy_analysis={"error": f"Analysis failed: {str(e)}"},
            processing_time=processing_time
        )

@agent.on_rest_post("/analyze_text", TextAnalysisRequest, TextAnalysisResponse)
async def handle_text_analysis(ctx: Context, request: TextAnalysisRequest) -> TextAnalysisResponse:
    """Handle analysis of pre-transcribed text"""
    start_time = datetime.now()
    request_id = f"text_{int(start_time.timestamp())}"
    
    try:
        ctx.logger.info(f"📝 Received text analysis request: {len(request.text)} characters")
        
        # Get analysis options as dict
        analysis_opts = dict(request.analysis_options) if hasattr(request.analysis_options, 'items') else request.analysis_options
        ctx.logger.info(f"🔧 Analysis options: {analysis_opts}")
        
        # Analyze hesitancy patterns
        hesitancy_analysis = {}
        if analysis_opts.get("hesitancy_analysis", True):
            ctx.logger.info("🔍 Starting hesitancy analysis...")
            try:
                hesitancy_analysis = await analyzer.analyze_hesitancy_with_asi1(request.text)
                ctx.logger.info(f"✅ Hesitancy analysis completed: {type(hesitancy_analysis)}")
                ctx.logger.info(f"📊 Analysis result keys: {list(hesitancy_analysis.keys()) if isinstance(hesitancy_analysis, dict) else 'Not a dict'}")
            except Exception as e:
                ctx.logger.error(f"❌ Hesitancy analysis failed: {str(e)}")
                hesitancy_analysis = {"error": f"Hesitancy analysis failed: {str(e)}"}
        
        # Extract keywords
        keywords = []
        if analysis_opts.get("keyword_extraction", True):
            ctx.logger.info("🔑 Starting keyword extraction...")
            try:
                keywords = analyzer.extract_simple_keywords(request.text)
                ctx.logger.info(f"✅ Keyword extraction completed: {len(keywords)} keywords")
                ctx.logger.info(f"🔤 Keywords type: {type(keywords)}")
            except Exception as e:
                ctx.logger.error(f"❌ Keyword extraction failed: {str(e)}")
                keywords = []
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log response before creating it
        ctx.logger.info(f"📤 Creating response:")
        ctx.logger.info(f"   request_id: {request_id} (type: {type(request_id)})")
        ctx.logger.info(f"   success: True (type: {type(True)})")
        ctx.logger.info(f"   hesitancy_analysis type: {type(hesitancy_analysis)}")
        ctx.logger.info(f"   keywords type: {type(keywords)}")
        ctx.logger.info(f"   processing_time: {processing_time} (type: {type(processing_time)})")
        
        response = TextAnalysisResponse(
            request_id=request_id,
            success=True,
            hesitancy_analysis=hesitancy_analysis,
            keywords=keywords,
            processing_time=processing_time,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        ctx.logger.info(f"✅ Response created successfully")
        
        # Try converting to dict to debug
        try:
            response_dict = response.dict()
            ctx.logger.info(f"📋 Response as dict: {response_dict}")
            ctx.logger.info(f"📋 Response dict type: {type(response_dict)}")
        except Exception as e:
            ctx.logger.error(f"❌ Failed to convert response to dict: {str(e)}")
        
        return response
        
    except Exception as e:
        ctx.logger.error(f"❌ Text analysis failed: {str(e)}")
        ctx.logger.error(f"❌ Exception type: {type(e)}")
        import traceback
        ctx.logger.error(f"❌ Traceback: {traceback.format_exc()}")
        processing_time = (datetime.now() - start_time).total_seconds()
        
        ctx.logger.info(f"📤 Creating error response:")
        ctx.logger.info(f"   request_id: {request_id} (type: {type(request_id)})")
        ctx.logger.info(f"   success: False (type: {type(False)})")
        ctx.logger.info(f"   processing_time: {processing_time} (type: {type(processing_time)})")
        
        return TextAnalysisResponse(
            request_id=request_id,
            success=False,
            hesitancy_analysis={"error": f"Analysis failed: {str(e)}"},
            keywords=[],
            processing_time=processing_time,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

@agent.on_rest_post("/analyze_covid", CovidAnalysisRequest, CovidAnalysisResponse)
async def handle_covid_analysis(ctx: Context, request: CovidAnalysisRequest) -> CovidAnalysisResponse:
    """Handle COVID voice analysis from audio files"""
    start_time = datetime.now()
    request_id = f"covid_{int(start_time.timestamp())}"
    
    try:
        ctx.logger.info(f"🦠 Received COVID analysis request: {request.filename}")
        
        # Decode base64 audio data
        try:
            audio_data = base64.b64decode(request.audio_data)
        except Exception as e:
            return CovidAnalysisResponse(
                request_id=request_id,
                success=False,
                covid_analysis={"error": f"Invalid base64 audio data: {str(e)}"},
                processing_time=0.0
            )
        
        # Get audio metadata
        audio_metadata = {
            "filename": request.filename,
            "size_bytes": len(audio_data),
            "format": request.filename.split('.')[-1].lower() if '.' in request.filename else 'mp3',
            "content_type": request.content_type
        }
        
        # Analyze for COVID indicators
        ctx.logger.info("🦠 Analyzing voice for COVID indicators...")
        covid_analysis = analyzer.detect_covid_from_voice(audio_data)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        ctx.logger.info(f"✅ COVID analysis completed in {processing_time:.2f} seconds")
        
        return CovidAnalysisResponse(
            request_id=request_id,
            success=True,
            covid_analysis=covid_analysis,
            processing_time=processing_time,
            audio_metadata=audio_metadata
        )
        
    except Exception as e:
        ctx.logger.error(f"❌ COVID analysis failed: {str(e)}")
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return CovidAnalysisResponse(
            request_id=request_id,
            success=False,
            covid_analysis={"error": f"Analysis failed: {str(e)}"},
            processing_time=processing_time
        )

@agent.on_rest_get("/health", HealthResponse)
async def handle_health_check(ctx: Context) -> HealthResponse:
    """Health check endpoint"""
    capabilities = [
        "Audio transcription (MP3, WAV, M4A, OGG)",
        "AI-powered vaccine hesitancy analysis",
        "Keyword extraction",
        "Text-only analysis"
    ]
    
    if analyzer.covid_model:
        capabilities.append("COVID voice detection (separate endpoint)")
    
    return HealthResponse(
        status="healthy",
        agent_name="Vaccine Hesitancy Voice Analyzer",
        capabilities=capabilities
    )

if __name__ == "__main__":
    print("""
🎤 Starting Vaccine Hesitancy Voice Analyzer Agent...

📋 Capabilities:
   • Audio transcription (MP3, WAV, M4A, OGG)
   • Vaccine hesitancy pattern analysis
   • Keyword extraction with timestamps
   • Optional COVID voice detection
   • Text-only analysis mode

📡 REST Endpoints:
   • POST /analyze_audio - Upload audio files for hesitancy analysis
   • POST /analyze_text  - Analyze pre-transcribed text
   • POST /analyze_covid - COVID voice detection from audio
   • GET  /health        - Health and capability check

📊 Analysis Features:
   • AI-powered hesitancy analysis with 6 categories
   • Intelligent understanding of context and nuance
   • Hesitancy scoring (0-100%) and level classification
   • Extracts supporting quotes and evidence

🔧 File Support:
   • MP3 files (primary format)
   • WAV, M4A, OGG formats
   • File size: 0.5-3MB (1-3 minute audio)
   • Base64 upload via REST API

🛑 Stop with Ctrl+C
    """)
    agent.run()
