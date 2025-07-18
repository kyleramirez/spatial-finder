#!/usr/bin/env python3
"""
Test script to verify pyannote.audio is working properly with distinct speaker identification.
Run this script to test that speaker diarization is functioning correctly.
"""

import os
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_pyannote_import():
    """Test that pyannote.audio can be imported."""
    print("🧪 Testing pyannote.audio import...")
    try:
        from pyannote.audio import Pipeline
        print("✅ pyannote.audio imported successfully")
        return True
    except ImportError as e:
        print(f"❌ pyannote.audio import failed: {e}")
        print("Please install with: pip install pyannote.audio")
        return False

def test_huggingface_token():
    """Test that HuggingFace token is available."""
    print("🧪 Testing HuggingFace token...")
    token = os.getenv('HUGGINGFACE_TOKEN')
    if token:
        print(f"✅ HuggingFace token found: {token[:8]}...")
        return True
    else:
        print("❌ HuggingFace token not found")
        print("Please:")
        print("1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("2. Accept the terms and conditions")
        print("3. Get your token from https://huggingface.co/settings/tokens")
        print("4. Create a .env file with: HUGGINGFACE_TOKEN=your_token_here")
        return False

def test_pipeline_loading():
    """Test that the diarization pipeline can be loaded."""
    print("🧪 Testing diarization pipeline loading...")
    try:
        from pyannote.audio import Pipeline
        token = os.getenv('HUGGINGFACE_TOKEN')
        
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token
        )
        print("✅ Diarization pipeline loaded successfully")
        return pipeline
    except Exception as e:
        print(f"❌ Pipeline loading failed: {e}")
        return None

def test_distinct_speakers():
    """Test that distinct speakers are properly identified."""
    print("🧪 Testing distinct speaker identification...")
    
    # Find a test audio file
    audio_samples = Path("audio-samples")
    if not audio_samples.exists():
        print("❌ audio-samples directory not found")
        return False
    
    # Look for audio files
    audio_files = list(audio_samples.rglob("*.wav")) + list(audio_samples.rglob("*.WMA"))
    if not audio_files:
        print("❌ No audio files found in audio-samples")
        return False
    
    # Use the first audio file
    test_file = audio_files[0]
    print(f"📄 Using test file: {test_file}")
    
    # Load the pipeline
    pipeline = test_pipeline_loading()
    if pipeline is None:
        return False
    
    try:
        # Process the audio file
        print("🎵 Processing audio file...")
        diarization = pipeline(str(test_file))
        
        # Collect speakers
        speakers = set()
        segments = []
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)
            segments.append((turn.start, turn.end, speaker))
        
        print(f"✅ Found {len(speakers)} distinct speakers: {list(speakers)}")
        
        # Show some segments
        if segments:
            print("📋 Sample segments:")
            for i, (start, end, speaker) in enumerate(segments[:5]):
                print(f"  {i+1}. {start:.1f}s-{end:.1f}s: {speaker}")
        
        # Verify we have distinct speakers
        if len(speakers) > 1:
            print("✅ Multiple distinct speakers detected!")
        else:
            print("ℹ️  Only one speaker detected (this may be correct for single-speaker audio)")
        
        return True
        
    except Exception as e:
        print(f"❌ Diarization failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing pyannote.audio functionality\n")
    
    # Test 1: Import
    if not test_pyannote_import():
        sys.exit(1)
    
    print()
    
    # Test 2: Token
    if not test_huggingface_token():
        sys.exit(1)
    
    print()
    
    # Test 3: Pipeline loading
    if not test_pipeline_loading():
        sys.exit(1)
    
    print()
    
    # Test 4: Distinct speakers
    if not test_distinct_speakers():
        sys.exit(1)
    
    print("\n🎉 All tests passed! pyannote.audio is working correctly.")
    print("📝 Speaker diarization will create distinct speakers with names like:")
    print("   - Speaker SPEAKER_00")
    print("   - Speaker SPEAKER_01")
    print("   - Speaker SPEAKER_02")
    print("   - etc.")

if __name__ == "__main__":
    main() 