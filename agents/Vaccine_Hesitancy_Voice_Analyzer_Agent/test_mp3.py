#!/usr/bin/env python3
"""
Quick MP3 tester for the voice analyzer agent
Usage: python test_mp3.py /path/to/your/file.mp3
"""

import base64
import requests
import sys
import json

def test_mp3_file(file_path):
    """Test an MP3 file with the voice analyzer"""
    
    # Read and encode the file
    with open(file_path, 'rb') as f:
        audio_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Extract filename
    filename = file_path.split('/')[-1]
    
    # Create request
    payload = {
        "audio_data": audio_data,
        "filename": filename,
        "content_type": "audio/mpeg",
        "analysis_options": {
            "hesitancy_analysis": True,
            "keyword_extraction": True
        }
    }
    
    print(f"🎤 Testing file: {filename}")
    print(f"📏 File size: {len(audio_data)/1024:.1f}KB (base64)")
    print("🔄 Sending request (this may take 1-3 minutes for transcription + AI analysis)...")
    
    # Send request
    try:
        response = requests.post(
            "http://localhost:8003/analyze_audio",
            json=payload,
            timeout=180  # 3 minutes for longer audio files
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"📝 Transcript: {result.get('transcript', '')[:200]}...")
            print(f"📊 Hesitancy Score: {result.get('hesitancy_analysis', {}).get('hesitancy_score', 'N/A')}")
            print(f"⏱️ Processing Time: {result.get('processing_time', 'N/A')}s")
            
            # Pretty print the full result
            print("\n📋 Full Response:")
            print(json.dumps(result, indent=2))
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Is the agent running on port 8003?")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_mp3.py /path/to/file.mp3")
        print("Example: python test_mp3.py /Users/davelradindra/Downloads/vaccine_clip.mp3")
        sys.exit(1)
    
    file_path = sys.argv[1]
    test_mp3_file(file_path) 