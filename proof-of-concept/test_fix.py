#!/usr/bin/env python3
"""
Test script to verify MPS fallback works correctly.
"""

import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import after loading environment variables
from main import AudioToolsCLI

def test_mps_fallback():
    """Test that MPS fallback works correctly."""
    print("🧪 Testing MPS fallback functionality...")
    
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Create CLI instance - this will trigger device selection and model loading
            cli = AudioToolsCLI()
            
            print(f"✅ Device selected: {cli.processor.device}")
            print(f"✅ Models loaded successfully")
            
            # Test that we can load models without crashing
            cli.processor.load_models()
            
            print("✅ Model loading test passed!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            raise
        finally:
            os.chdir(original_cwd)

if __name__ == "__main__":
    test_mps_fallback() 