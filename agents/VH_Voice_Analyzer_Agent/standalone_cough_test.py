#!/usr/bin/env python3
"""
Standalone cough detection test
Tests the cough detection functionality directly without REST API
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vh_voice_analyzer_rest_agent import VoiceAnalyzer
import time

def test_cough_detection_standalone():
    """Test cough detection directly using the VoiceAnalyzer class"""
    
    # Configuration
    cough_file_path = "/Users/davelradindra/Downloads/covid_positive2.wav"
    
    print("🎤 Standalone Cough Detection Test")
    print("=" * 50)
    
    # Initialize analyzer
    print("🔧 Initializing VoiceAnalyzer...")
    analyzer = VoiceAnalyzer()
    
    # Check if COVID model loaded
    if not analyzer.covid_model:
        print("❌ COVID model failed to load!")
        print("💡 This might be due to:")
        print("   - Missing huggingface_hub dependency")
        print("   - Network connection issues")
        print("   - Model download problems")
        return False
    
    print("✅ COVID model loaded successfully")
    print(f"📊 Model components: {list(analyzer.covid_model.keys())}")
    
    # Load audio file
    print(f"\n📁 Loading audio file: {cough_file_path}")
    
    if not os.path.exists(cough_file_path):
        print(f"❌ Audio file not found: {cough_file_path}")
        return False
    
    try:
        with open(cough_file_path, 'rb') as f:
            audio_data = f.read()
        
        print(f"✅ Audio file loaded: {len(audio_data):,} bytes")
        
    except Exception as e:
        print(f"❌ Error loading audio file: {e}")
        return False
    
    # Test cough detection
    print(f"\n🦠 Testing cough detection...")
    start_time = time.time()
    
    result = analyzer.detect_covid_from_voice(audio_data)
    
    processing_time = time.time() - start_time
    print(f"⏱️ Processing completed in {processing_time:.2f} seconds")
    
    if result:
        print(f"\n✅ Cough detection successful!")
        print(f"🦠 Results:")
        print(f"   Predicted class: {result['predicted_class']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Note: {result['note']}")
        
        print(f"\n📊 Class Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"   {class_name}: {prob:.3f} ({prob*100:.1f}%)")
        
        return True
    else:
        print(f"❌ Cough detection failed!")
        return False

def main():
    """Main function"""
    success = test_cough_detection_standalone()
    
    if success:
        print("\n✅ Standalone test completed successfully!")
    else:
        print("\n❌ Standalone test failed!")
        print("\n💡 Troubleshooting tips:")
        print("   1. Make sure all dependencies are installed:")
        print("      pip install librosa pandas numpy scikit-learn huggingface_hub")
        print("   2. Check your internet connection for model download")
        print("   3. Verify the audio file path is correct")
        print("   4. Check the audio file format (MP3 should work)")
    
    print("\n" + "=" * 50)
    print("🏁 Test completed")

if __name__ == "__main__":
    main() 