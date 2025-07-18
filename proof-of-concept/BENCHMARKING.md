# Benchmarking Guide for Audible Tools

This guide explains how to benchmark CPU vs GPU performance and understand the performance characteristics of the audio processing pipeline.

## Quick Start

### Basic Whisper Benchmark
```bash
python simple_benchmark.py
```

### Full System Benchmark
```bash
python benchmark.py --max-files 3
```

## Understanding the Results

### Sample Output (Simple Benchmark)
```
🚀 Simple Whisper Benchmark
==================================================
📄 Test file: DS400288.WMA (832.3 KB)
✓ Apple MPS detected and working
🖥️  Available devices: ['cpu', 'mps']

🎯 Testing Whisper on CPU
✓ CPU: 31.91s transcription, 0.00x real-time

🎯 Testing Whisper on MPS
✗ MPS: Error - Could not run 'aten::_sparse_coo_tensor_with_dims_and_tensors'...

📊 BENCHMARK RESULTS
+----------+------------------+---------------------+-------------+---------------+-----------+
| Device   |   Model Load (s) |   Transcription (s) |   Total (s) | Real-time Factor | Speedup   |
+==========+==================+=====================+=============+===============+===========+
| CPU      |             0.56 |               31.91 |       32.47 | 0.00x         | 1.00x     |
+----------+------------------+---------------------+-------------+---------------+-----------+
```

### Key Metrics Explained

1. **Model Load Time**: Time to load Whisper model into memory
2. **Transcription Time**: Time to process audio and generate transcript
3. **Real-time Factor**: How much faster than real-time playback (lower is better)
4. **Speedup**: Performance improvement over CPU baseline

## Performance Analysis

### CPU Performance Characteristics

**From our 832KB test file:**
- **Processing Time**: ~32 seconds
- **Real-time Factor**: Very slow (much slower than real-time)
- **Memory Usage**: ~2GB during processing
- **Throughput**: ~0.026 MB/s

### Expected GPU Performance (when working)

**Apple M1/M2/M3 (MPS):**
- **Expected Speedup**: 2-4x faster than CPU
- **Processing Time**: ~8-16 seconds for same file
- **Memory Usage**: Similar to CPU
- **Real-time Factor**: Much closer to real-time

**NVIDIA GPU (CUDA):**
- **Expected Speedup**: 3-8x faster than CPU
- **Processing Time**: ~4-11 seconds for same file
- **Memory Usage**: GPU VRAM + system RAM
- **Real-time Factor**: Often faster than real-time

## Current GPU Issues

### MPS (Apple Silicon) Issue
The current PyTorch MPS backend fails with sparse tensor operations:
```
Could not run 'aten::_sparse_coo_tensor_with_dims_and_tensors' with arguments from the 'SparseMPS' backend
```

This affects:
- ✅ **Whisper transcription**: Should work on MPS
- ❌ **Sentence transformers**: Fails on MPS, falls back to CPU
- ✅ **Database operations**: CPU-only anyway

### Workaround Strategy

The code implements a hybrid approach:
1. **Try MPS/CUDA first** for Whisper (main compute load)
2. **Fall back to CPU** for sentence transformers when MPS fails
3. **Use CPU** for database operations (not GPU-accelerated)

## Benchmarking Different Scenarios

### 1. Small Files (< 100KB)
```bash
# Test with smallest files
python simple_benchmark.py
```

**Expected results:**
- Model loading dominates processing time
- GPU advantage minimal due to overhead

### 2. Medium Files (100KB - 1MB)
```bash
# Test with medium files
python benchmark.py --max-files 1
```

**Expected results:**
- GPU advantage becomes apparent
- Processing time scales with file size

### 3. Large Files (> 1MB)
```bash
# Test with larger files
find audio-samples -name "*.WMA" -size +1000k | head -1 | xargs python simple_benchmark.py --files
```

**Expected results:**
- Maximum GPU benefit
- Memory usage becomes important

## Performance Optimization Tips

### 1. Model Selection
Different Whisper models have different performance characteristics:

```python
# Faster but less accurate
model = whisper.load_model("tiny", device=device)

# Slower but more accurate  
model = whisper.load_model("large", device=device)
```

### 2. Batch Processing
Process multiple files to amortize model loading:
```bash
# Process multiple files at once
python benchmark.py --files audio1.wma audio2.wma audio3.wma
```

### 3. Memory Management
For large files, monitor memory usage:
```bash
# Monitor memory during processing
htop  # or Activity Monitor on Mac
```

## Interpreting Results

### Good Performance Indicators
- **Real-time factor < 1.0**: Processing faster than playback
- **GPU speedup > 2x**: Significant acceleration
- **Model load time < 5s**: Efficient model loading

### Performance Issues
- **Real-time factor > 5.0**: Very slow processing
- **High memory usage**: May indicate memory leaks
- **GPU fallback**: Check for compatibility issues

## Troubleshooting Performance

### 1. Slow Processing
```bash
# Check if GPU is being used
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"

# Monitor GPU usage
nvidia-smi  # for NVIDIA
```

### 2. Memory Issues
```bash
# Check available memory
free -h  # Linux
vm_stat  # Mac
```

### 3. Model Loading Issues
```bash
# Clear model cache
rm -rf ~/.cache/whisper/
```

## Creating Custom Benchmarks

### Basic Custom Benchmark
```python
import time
import whisper

def benchmark_custom():
    device = "cpu"  # or "cuda", "mps"
    
    start = time.time()
    model = whisper.load_model("base", device=device)
    load_time = time.time() - start
    
    start = time.time()
    result = model.transcribe("your_audio.wma")
    transcribe_time = time.time() - start
    
    print(f"Load: {load_time:.2f}s, Transcribe: {transcribe_time:.2f}s")
```

### Advanced Benchmark with Memory Monitoring
```python
import psutil
import time

def benchmark_with_memory():
    process = psutil.Process()
    
    # Memory before
    mem_before = process.memory_info().rss / 1024 / 1024
    
    # Your processing code here
    
    # Memory after
    mem_after = process.memory_info().rss / 1024 / 1024
    
    print(f"Memory usage: {mem_after - mem_before:.1f} MB")
```

## Expected Future Performance

Once GPU compatibility issues are resolved:

### Apple Silicon (M1/M2/M3)
- **Whisper**: 2-4x speedup
- **Overall**: 1.5-2x speedup (limited by CPU operations)
- **Memory**: Similar usage, better efficiency

### NVIDIA GPU
- **Whisper**: 3-8x speedup
- **Overall**: 2-4x speedup
- **Memory**: GPU VRAM usage, potential for larger models

### Intel GPU (future)
- **Arc GPUs**: 2-5x speedup expected
- **Integrated**: 1.5-2x speedup expected

## Conclusion

While GPU acceleration is currently limited due to compatibility issues, the benchmarking tools provide valuable insights into:

1. **Current CPU performance**: Baseline for comparison
2. **Bottlenecks**: Where optimization would help most
3. **Scaling characteristics**: How performance changes with file size
4. **Resource usage**: Memory and compute requirements

Use these tools to:
- **Monitor performance** as you process files
- **Compare different approaches** (model sizes, batch sizes)
- **Identify optimal configurations** for your use case
- **Track improvements** as GPU support is enhanced 