#!/usr/bin/env python3
"""
Simple benchmark script to compare CPU vs GPU performance for Whisper processing.
This focuses on Whisper transcription performance which is the most compute-intensive part.
"""

import os
import time
import tempfile
import warnings
from pathlib import Path
import torch
import whisper
from tabulate import tabulate

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")

def get_available_devices():
    """Get list of available devices."""
    devices = ['cpu']
    
    if torch.cuda.is_available():
        try:
            test_tensor = torch.tensor([1.0], device="cuda")
            devices.append('cuda')
            print("✓ CUDA GPU detected and working")
        except Exception as e:
            print(f"✗ CUDA available but not working: {e}")
    
    if torch.backends.mps.is_available():
        try:
            test_tensor = torch.tensor([1.0], device="mps")
            result = test_tensor * 2
            devices.append('mps')
            print("✓ Apple MPS detected and working")
        except Exception as e:
            print(f"✗ MPS available but not working: {e}")
    
    return devices

def benchmark_whisper_transcription(audio_file: Path, device: str) -> dict:
    """Benchmark Whisper transcription on a specific device."""
    print(f"\n🎯 Testing Whisper on {device.upper()}")
    
    result = {
        'device': device,
        'file_name': audio_file.name,
        'file_size_mb': audio_file.stat().st_size / (1024 * 1024),
        'model_load_time': 0.0,
        'transcription_time': 0.0,
        'total_time': 0.0,
        'audio_duration': 0.0,
        'real_time_factor': 0.0,
        'success': False,
        'error': None
    }
    
    try:
        # Time model loading
        model_start = time.time()
        model = whisper.load_model("base", device=device)
        result['model_load_time'] = time.time() - model_start
        
        # Time transcription
        transcription_start = time.time()
        transcription = model.transcribe(str(audio_file))
        result['transcription_time'] = time.time() - transcription_start
        
        # Calculate metrics
        result['total_time'] = result['model_load_time'] + result['transcription_time']
        result['audio_duration'] = transcription.get('duration', 0)
        
        if result['audio_duration'] > 0:
            result['real_time_factor'] = result['transcription_time'] / result['audio_duration']
        
        result['success'] = True
        
        print(f"✓ {device.upper()}: {result['transcription_time']:.2f}s transcription, {result['real_time_factor']:.2f}x real-time")
        
        # Clean up model to free memory
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
        elif device == 'mps':
            torch.mps.empty_cache()
            
    except Exception as e:
        result['error'] = str(e)
        result['success'] = False
        print(f"✗ {device.upper()}: Error - {e}")
    
    return result

def find_test_audio_file() -> Path:
    """Find a suitable test audio file."""
    audio_dir = Path("audio-samples")
    if not audio_dir.exists():
        raise FileNotFoundError("audio-samples directory not found")
    
    # Find WMA files
    wma_files = list(audio_dir.rglob("*.WMA"))
    if not wma_files:
        raise FileNotFoundError("No WMA files found in audio-samples")
    
    # Sort by size and pick a medium-sized file
    wma_files.sort(key=lambda f: f.stat().st_size)
    
    if len(wma_files) >= 3:
        # Pick the middle-sized file
        return wma_files[len(wma_files) // 2]
    else:
        # Pick the first available file
        return wma_files[0]

def main():
    """Main benchmark function."""
    print("🚀 Simple Whisper Benchmark")
    print("=" * 50)
    
    # Find test file
    try:
        test_file = find_test_audio_file()
        print(f"📄 Test file: {test_file.name} ({test_file.stat().st_size / 1024:.1f} KB)")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Get available devices
    devices = get_available_devices()
    print(f"🖥️  Available devices: {devices}")
    
    # Run benchmarks
    results = []
    for device in devices:
        result = benchmark_whisper_transcription(test_file, device)
        results.append(result)
        
        # Add a delay between tests to let GPU cool down
        if device in ['cuda', 'mps']:
            time.sleep(2)
    
    # Print results
    print("\n" + "=" * 80)
    print("📊 BENCHMARK RESULTS")
    print("=" * 80)
    
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        # Create comparison table
        headers = ["Device", "Model Load (s)", "Transcription (s)", "Total (s)", "Real-time Factor", "Speedup"]
        rows = []
        
        cpu_time = None
        for result in successful_results:
            if result['device'] == 'cpu':
                cpu_time = result['transcription_time']
                break
        
        for result in successful_results:
            speedup = f"{cpu_time / result['transcription_time']:.2f}x" if cpu_time and result['transcription_time'] > 0 else "N/A"
            
            rows.append([
                result['device'].upper(),
                f"{result['model_load_time']:.2f}",
                f"{result['transcription_time']:.2f}",
                f"{result['total_time']:.2f}",
                f"{result['real_time_factor']:.2f}x",
                speedup
            ])
        
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        
        # Show performance summary
        if len(successful_results) > 1:
            fastest = min(successful_results, key=lambda r: r['transcription_time'])
            print(f"\n🏆 Fastest device: {fastest['device'].upper()}")
            print(f"   📊 Performance: {fastest['transcription_time']:.2f}s ({fastest['real_time_factor']:.2f}x real-time)")
            
            if cpu_time and fastest['device'] != 'cpu':
                speedup = cpu_time / fastest['transcription_time']
                print(f"   ⚡ Speedup over CPU: {speedup:.2f}x")
    
    # Show failed results
    failed_results = [r for r in results if not r['success']]
    if failed_results:
        print("\n❌ Failed tests:")
        for result in failed_results:
            print(f"   {result['device'].upper()}: {result['error']}")

if __name__ == '__main__':
    main() 