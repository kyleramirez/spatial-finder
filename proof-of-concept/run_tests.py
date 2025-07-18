#!/usr/bin/env python3
"""
Simple test runner for Audible Tools
Makes it easy to run different types of tests.
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"\n{description} completed in {duration:.2f} seconds")
    
    if result.returncode == 0:
        print("✓ PASSED")
    else:
        print("✗ FAILED")
        return False
    
    return True


def check_environment():
    """Check that the environment is set up correctly."""
    print("Checking environment...")
    
    # Check if we're in virtual environment
    in_venv = (
        hasattr(sys, 'real_prefix') or 
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    )
    
    if in_venv:
        print("✓ Running in virtual environment")
    else:
        print("⚠ Not running in virtual environment")
        print("  Run: source virtualenv/bin/activate")
    
    # Check if audio files exist
    audio_files = list(Path("audio-samples").rglob("*.WMA"))
    if audio_files:
        print(f"✓ Found {len(audio_files)} audio files for testing")
    else:
        print("⚠ No audio files found in audio-samples/")
    
    # Check required modules
    required_modules = ['click', 'duckdb', 'torch', 'whisper', 'librosa']
    missing = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module} installed")
        except ImportError:
            missing.append(module)
            print(f"✗ {module} not installed")
    
    if missing:
        print(f"\nMissing modules: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True


def main():
    """Main test runner."""
    print("Audible Tools Test Runner")
    print("=" * 40)
    
    # Check environment
    if not check_environment():
        print("\nEnvironment check failed. Please fix issues above.")
        sys.exit(1)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run Audible Tools tests')
    parser.add_argument('--quick', action='store_true',
                       help='Run only quick unit tests')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration tests with real audio files')
    parser.add_argument('--performance', action='store_true',
                       help='Run performance benchmarks')
    parser.add_argument('--all', action='store_true',
                       help='Run all tests (default)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Default to running all tests
    if not any([args.quick, args.integration, args.performance]):
        args.all = True
    
    success = True
    
    # Run basic functionality test
    if args.quick or args.all:
        cmd = "python audible-tools.py --help"
        if not run_command(cmd, "Basic CLI functionality test"):
            success = False
    
    # Run unit tests
    if args.quick or args.all:
        cmd = "python test_audible_tools.py"
        if args.verbose:
            cmd += " --verbose"
        if not run_command(cmd, "Unit tests"):
            success = False
    
    # Run integration tests
    if args.integration or args.all:
        # First reset any existing database
        if Path("audible_tools.db").exists():
            Path("audible_tools.db").unlink()
        
        # Test with a small audio file
        audio_files = list(Path("audio-samples").rglob("*.WMA"))
        if audio_files:
            test_file = min(audio_files, key=lambda f: f.stat().st_size)
            cmd = f'python audible-tools.py add "{test_file}"'
            if not run_command(cmd, f"Integration test with {test_file.name}"):
                success = False
            
            # Test status
            cmd = "python audible-tools.py status"
            if not run_command(cmd, "Status check"):
                success = False
            
            # Test export
            cmd = f'python audible-tools.py export "{test_file}"'
            if not run_command(cmd, "Export test"):
                success = False
    
    # Run performance tests
    if args.performance or args.all:
        cmd = "python test_audible_tools.py --performance"
        if not run_command(cmd, "Performance benchmarks"):
            success = False
    
    # Final summary
    print(f"\n{'='*60}")
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("The Audible Tools implementation is working correctly.")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the output above for details.")
    print('='*60)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main() 