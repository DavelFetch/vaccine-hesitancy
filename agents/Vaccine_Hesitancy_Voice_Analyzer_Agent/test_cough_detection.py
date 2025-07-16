#!/usr/bin/env python3
"""
Test script for cough detection functionality
Tests the /analyze_covid endpoint with a local MP3 file
"""

import base64
import requests
import json
import time
import os
from pathlib import Path

# Configuration
AGENT_URL = "http://localhost:8003"
COUGH_FILE_PATH = "/Users/davelradindra/Downloads/covid_cough.mp3"

def load_and_encode_audio(file_path: str) -> tuple[str, str]:
    """Load audio file and encode as base64"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            audio_data = f.read()
        
        # Encode to base64
        encoded_audio = base64.b64encode(audio_data).decode('utf-8')
        
        # Get filename
        filename = Path(file_path).name
        
        print(f"✅ Loaded audio file: {filename}")
        print(f"📊 File size: {len(audio_data):,} bytes")
        print(f"📊 Base64 size: {len(encoded_audio):,} characters")
        
        return encoded_audio, filename
        
    except Exception as e:
        print(f"❌ Error loading audio file: {e}")
        return None, None

def test_health_endpoint():
    """Test the health endpoint to make sure agent is running"""
    try:
        print("🏥 Testing health endpoint...")
        response = requests.get(f"{AGENT_URL}/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Agent is healthy: {health_data['status']}")
            print(f"🤖 Agent: {health_data['agent_name']}")
            print(f"🔧 Capabilities: {', '.join(health_data['capabilities'])}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to agent. Make sure it's running on port 8003")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_cough_detection(encoded_audio: str, filename: str):
    """Test the cough detection endpoint"""
    try:
        print(f"🦠 Testing cough detection with {filename}...")
        
        # Prepare request payload
        payload = {
            "audio_data": encoded_audio,
            "filename": filename,
            "content_type": "audio/mpeg"
        }
        
        # Make request
        start_time = time.time()
        response = requests.post(
            f"{AGENT_URL}/analyze_covid",
            json=payload,
            timeout=120  # 2 minutes timeout
        )
        request_time = time.time() - start_time
        
        print(f"⏱️ Request completed in {request_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Cough detection successful!")
            print(f"📋 Request ID: {result['request_id']}")
            print(f"⏱️ Processing time: {result['processing_time']:.2f} seconds")
            
            # Display COVID analysis results
            if result['covid_analysis']:
                covid_data = result['covid_analysis']
                print(f"\n🦠 COVID Analysis Results:")
                print(f"   Predicted class: {covid_data['predicted_class']}")
                print(f"   Confidence: {covid_data['confidence']:.3f}")
                print(f"   Note: {covid_data['note']}")
                
                print(f"\n📊 Class Probabilities:")
                for class_name, prob in covid_data['probabilities'].items():
                    print(f"   {class_name}: {prob:.3f} ({prob*100:.1f}%)")
            else:
                print("⚠️ No COVID analysis results returned")
            
            # Display audio metadata
            if result['audio_metadata']:
                metadata = result['audio_metadata']
                print(f"\n🎵 Audio Metadata:")
                print(f"   Filename: {metadata['filename']}")
                print(f"   Size: {metadata['size_bytes']:,} bytes")
                print(f"   Format: {metadata['format']}")
                print(f"   Content Type: {metadata['content_type']}")
            
            return True
            
        else:
            print(f"❌ Cough detection failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"❌ Error details: {error_data}")
            except:
                print(f"❌ Response text: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out. The audio file might be too large or processing is slow.")
        return False
    except Exception as e:
        print(f"❌ Cough detection error: {e}")
        return False

def main():
    """Main test function"""
    print("🎤 Cough Detection Test Script")
    print("=" * 50)
    
    # Step 1: Test health endpoint
    if not test_health_endpoint():
        print("\n❌ Cannot proceed - agent is not running or not healthy")
        print("💡 Make sure to run: python vh_voice_analyzer_rest_agent.py")
        return
    
    # Step 2: Load audio file
    print(f"\n📁 Loading audio file: {COUGH_FILE_PATH}")
    encoded_audio, filename = load_and_encode_audio(COUGH_FILE_PATH)
    
    if not encoded_audio:
        print("❌ Cannot proceed - failed to load audio file")
        return
    
    # Step 3: Test cough detection
    print(f"\n🦠 Testing cough detection...")
    success = test_cough_detection(encoded_audio, filename)
    
    if success:
        print("\n✅ Cough detection test completed successfully!")
    else:
        print("\n❌ Cough detection test failed!")
    
    print("\n" + "=" * 50)
    print("🏁 Test completed")

if __name__ == "__main__":
    main() 