#!/usr/bin/env python3
"""
Benchmarking script for Audible Tools
Compares performance between CPU and GPU processing.
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple
import warnings

import torch
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

# Import our modules
from main import AudioToolsCLI, AudioProcessor, AudioDatabase, get_device

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")

class BenchmarkResult:
    """Container for benchmark results."""
    def __init__(self, device: str, file_name: str, file_size: int, duration: float):
        self.device = device
        self.file_name = file_name
        self.file_size = file_size
        self.duration = duration
        self.processing_time = 0.0
        self.model_load_time = 0.0
        self.transcription_time = 0.0
        self.verbalizations = 0
        self.words_per_second = 0.0
        self.mb_per_second = 0.0
        self.success = False
        self.error_message = ""

    def calculate_metrics(self):
        """Calculate derived metrics."""
        if self.processing_time > 0:
            self.mb_per_second = (self.file_size / 1024 / 1024) / self.processing_time
            if self.duration > 0:
                self.words_per_second = self.verbalizations / self.duration

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'device': self.device,
            'file_name': self.file_name,
            'file_size_mb': round(self.file_size / 1024 / 1024, 2),
            'duration_seconds': self.duration,
            'processing_time': round(self.processing_time, 2),
            'model_load_time': round(self.model_load_time, 2),
            'transcription_time': round(self.transcription_time, 2),
            'verbalizations': self.verbalizations,
            'words_per_second': round(self.words_per_second, 2),
            'mb_per_second': round(self.mb_per_second, 2),
            'success': self.success,
            'error_message': self.error_message
        }

class AudioBenchmarker:
    """Benchmarking class for audio processing."""
    
    def __init__(self):
        self.results = []
        self.available_devices = self.detect_devices()
    
    def detect_devices(self) -> List[str]:
        """Detect available compute devices."""
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
        
        print(f"Available devices: {devices}")
        return devices
    
    def benchmark_file(self, file_path: Path, device: str, force_device: bool = False) -> BenchmarkResult:
        """Benchmark processing of a single file on a specific device."""
        result = BenchmarkResult(
            device=device,
            file_name=file_path.name,
            file_size=file_path.stat().st_size,
            duration=0.0
        )
        
        print(f"\nBenchmarking {file_path.name} on {device.upper()}...")
        
        # Create temporary directory for this test
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                # Force device if requested
                if force_device:
                    # Monkey patch the get_device function
                    import main
                    original_get_device = main.get_device
                    main.get_device = lambda: device
                
                # Create CLI and processor
                cli = AudioToolsCLI()
                processor = cli.processor
                
                # Time model loading
                model_load_start = time.time()
                processor.load_models()
                result.model_load_time = time.time() - model_load_start
                
                # Get audio duration
                metadata = processor.get_audio_metadata(str(file_path))
                result.duration = metadata.get('duration', 0.0)
                
                # Time full processing
                processing_start = time.time()
                
                # Add file to database
                cli.add_files([str(file_path)])
                
                result.processing_time = time.time() - processing_start
                
                # Check results
                status = cli.db.conn.execute(
                    "SELECT ingest_status FROM audible_files WHERE basename = ?",
                    (file_path.name,)
                ).fetchone()
                
                if status and status[0] == 'COMPLETE':
                    result.success = True
                    
                    # Count verbalizations
                    verbalizations = cli.db.conn.execute(
                        "SELECT COUNT(*) FROM verbalizations"
                    ).fetchone()[0]
                    result.verbalizations = verbalizations
                    
                    # Calculate metrics
                    result.calculate_metrics()
                    
                    print(f"✓ {device.upper()}: {result.processing_time:.2f}s, {result.verbalizations} verbalizations")
                else:
                    result.success = False
                    result.error_message = f"Processing failed with status: {status[0] if status else 'Unknown'}"
                    print(f"✗ {device.upper()}: Processing failed")
                
                # Restore original get_device function
                if force_device:
                    main.get_device = original_get_device
                
            except Exception as e:
                result.success = False
                result.error_message = str(e)
                print(f"✗ {device.upper()}: Error - {e}")
                
                # Restore original get_device function
                if force_device:
                    main.get_device = original_get_device
            
            finally:
                os.chdir(original_cwd)
        
        return result
    
    def benchmark_files(self, file_paths: List[Path], devices: List[str] = None) -> List[BenchmarkResult]:
        """Benchmark multiple files on multiple devices."""
        if devices is None:
            devices = self.available_devices
        
        results = []
        
        for file_path in file_paths:
            print(f"\n{'='*60}")
            print(f"Benchmarking: {file_path.name} ({file_path.stat().st_size / 1024:.1f} KB)")
            print('='*60)
            
            for device in devices:
                if device not in self.available_devices:
                    print(f"⚠ Skipping {device} (not available)")
                    continue
                
                result = self.benchmark_file(file_path, device, force_device=True)
                results.append(result)
        
        return results
    
    def compare_devices(self, results: List[BenchmarkResult]) -> Dict:
        """Compare performance across devices."""
        comparison = {}
        
        # Group results by file
        by_file = {}
        for result in results:
            if result.file_name not in by_file:
                by_file[result.file_name] = {}
            by_file[result.file_name][result.device] = result
        
        # Calculate speedup metrics
        for file_name, file_results in by_file.items():
            if 'cpu' in file_results:
                cpu_time = file_results['cpu'].processing_time
                
                for device, result in file_results.items():
                    if device != 'cpu' and result.success:
                        speedup = cpu_time / result.processing_time if result.processing_time > 0 else 0
                        comparison[f"{file_name}_{device}_speedup"] = speedup
        
        return comparison
    
    def print_results(self, results: List[BenchmarkResult]):
        """Print benchmark results in a nice format."""
        print(f"\n{'='*80}")
        print("BENCHMARK RESULTS")
        print('='*80)
        
        # Group by file
        by_file = {}
        for result in results:
            if result.file_name not in by_file:
                by_file[result.file_name] = []
            by_file[result.file_name].append(result)
        
        for file_name, file_results in by_file.items():
            print(f"\n📁 {file_name}")
            print("-" * 60)
            
            # Prepare table data
            headers = ["Device", "Status", "Time (s)", "Verbalizations", "MB/s", "Speedup"]
            rows = []
            
            cpu_time = None
            for result in file_results:
                if result.device == 'cpu':
                    cpu_time = result.processing_time
                    break
            
            for result in file_results:
                status = "✓" if result.success else "✗"
                speedup = f"{cpu_time / result.processing_time:.2f}x" if cpu_time and result.processing_time > 0 else "N/A"
                
                rows.append([
                    result.device.upper(),
                    status,
                    f"{result.processing_time:.2f}",
                    result.verbalizations,
                    f"{result.mb_per_second:.2f}",
                    speedup
                ])
            
            print(tabulate(rows, headers=headers, tablefmt="grid"))
            
            # Print errors if any
            for result in file_results:
                if not result.success and result.error_message:
                    print(f"❌ {result.device.upper()}: {result.error_message}")
    
    def save_results(self, results: List[BenchmarkResult], output_file: str):
        """Save results to JSON file."""
        data = {
            'benchmark_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'available_devices': self.available_devices,
            'results': [result.to_dict() for result in results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n📊 Results saved to {output_file}")

def find_test_files(audio_dir: Path, max_files: int = 3) -> List[Path]:
    """Find suitable test files for benchmarking."""
    audio_files = list(audio_dir.rglob("*.WMA"))
    
    if not audio_files:
        print("❌ No audio files found for benchmarking")
        return []
    
    # Sort by size and pick a range
    audio_files.sort(key=lambda f: f.stat().st_size)
    
    # Pick small, medium, and large files
    selected = []
    if len(audio_files) >= 3:
        selected.append(audio_files[0])  # Smallest
        selected.append(audio_files[len(audio_files) // 2])  # Medium
        selected.append(audio_files[-1])  # Largest
    else:
        selected = audio_files[:max_files]
    
    return selected

def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(description='Benchmark Audible Tools performance')
    parser.add_argument('--files', nargs='+', help='Specific files to benchmark')
    parser.add_argument('--devices', nargs='+', choices=['cpu', 'cuda', 'mps'], 
                       help='Devices to test (default: all available)')
    parser.add_argument('--output', default='benchmark_results.json', 
                       help='Output file for results')
    parser.add_argument('--max-files', type=int, default=3, 
                       help='Maximum number of files to test')
    
    args = parser.parse_args()
    
    # Initialize benchmarker
    benchmarker = AudioBenchmarker()
    
    # Find test files
    if args.files:
        test_files = [Path(f) for f in args.files]
    else:
        audio_dir = Path("audio-samples")
        if not audio_dir.exists():
            print("❌ audio-samples directory not found")
            return
        
        test_files = find_test_files(audio_dir, args.max_files)
    
    if not test_files:
        print("❌ No test files found")
        return
    
    print(f"🎯 Testing {len(test_files)} files on {len(benchmarker.available_devices)} devices")
    for f in test_files:
        print(f"   📄 {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    
    # Run benchmarks
    results = benchmarker.benchmark_files(test_files, args.devices)
    
    # Print results
    benchmarker.print_results(results)
    
    # Save results
    benchmarker.save_results(results, args.output)
    
    # Print summary
    successful_results = [r for r in results if r.success]
    if successful_results:
        print(f"\n🎉 Benchmarking complete!")
        print(f"   ✅ {len(successful_results)} successful tests")
        print(f"   ❌ {len(results) - len(successful_results)} failed tests")
        
        # Show best performance
        fastest = min(successful_results, key=lambda r: r.processing_time)
        print(f"   🏆 Fastest: {fastest.device.upper()} - {fastest.processing_time:.2f}s")

if __name__ == '__main__':
    main() 