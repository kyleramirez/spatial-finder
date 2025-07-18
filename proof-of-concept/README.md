# Audible Tools - Audio Processing and Transcription Tool

A command-line tool for processing audio files with automatic transcription and speaker diarization.

## Features

- **Audio File Ingestion**: Supports multiple audio formats (MP3, WAV, FLAC, M4A, AAC, OGG, WMA, AIFF, AU)
- **Automatic Transcription**: Uses OpenAI Whisper for speech-to-text conversion
- **Speaker Diarization**: Identifies and separates different speakers using pyannote.audio
- **Text Search**: Search through transcripts using natural language queries
- **Voice Management**: Rename and merge speaker identities
- **Export Functionality**: Generate SRT subtitle files with speaker labels
- **Database Storage**: Persistent storage using DuckDB with comprehensive schema

## Setup

1. **Create and activate virtual environment:**
   ```bash
   python -m venv virtualenv
   source virtualenv/bin/activate  # On Windows: virtualenv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Accept pyannote.audio terms (required for speaker diarization):**
   - Visit https://huggingface.co/pyannote/speaker-diarization-3.1
   - Accept the terms and conditions
   - Get your access token from https://huggingface.co/settings/tokens
   - Create a `.env` file in the `proof-of-concept` directory:
     ```bash
     echo "HUGGINGFACE_TOKEN=your_token_here" > .env
     ```
   - Replace `your_token_here` with your actual Hugging Face token


## Usage

### Basic Commands

**Add audio files to database:**
```bash
./audible-tools.py add ./audio-file.mp3
./audible-tools.py add -R ./audio-directory/  # Recursive
```

**Check processing status:**
```bash
./audible-tools.py status
```

**List files:**
```bash
./audible-tools.py ls
./audible-tools.py ls ./audio-directory/
```

### Voice Management

**List identified voices:**
```bash
./audible-tools.py voices list
```

**Rename a voice:**
```bash
./audible-tools.py voices rename 1 "Mike"
```

### Search and Discovery

**Search transcripts:**
```bash
./audible-tools.py search "dinner plans"
./audible-tools.py search "meeting" --limit 20
```

**Find all speech by a specific voice:**
```bash
./audible-tools.py voice "Mike"
./audible-tools.py voice "Speaker_1" --limit 15
```

### Export Transcripts

**Export as SRT files:**
```bash
./audible-tools.py export ./audio-file.mp3
./audible-tools.py export -R ./audio-directory/
```

### Database Management

**Remove files from database:**
```bash
./audible-tools.py rm ./audio-file.mp3
./audible-tools.py rm -R ./audio-directory/
```

**Reset entire database:**
```bash
./audible-tools.py reset
```

## File Processing Pipeline

1. **Ingestion**: Files are added to the database with metadata extraction
2. **Conversion**: Audio is converted to WAV format for processing
3. **Transcription**: Whisper generates text transcripts with timestamps
4. **Diarization**: pyannote.audio identifies and separates speakers
5. **Embedding**: Text is converted to embeddings for similarity search
6. **Storage**: All data is stored in DuckDB for fast querying

## Database Schema

The tool uses a comprehensive schema with tables for:
- **audible_files**: Audio file metadata and processing status
- **voices**: Speaker identities and management
- **verbalizations**: Transcribed speech segments with speaker attribution
- **nonverbal_labels**: Non-speech audio annotations
- **silents**: Silence detection results
- **audible_embeddings**: Audio similarity embeddings
- **exports**: Export job tracking

## Output Files

- **Transcripts**: `original-file.exported_YYYYMMDD_HHMMSS.srt`
- **Database**: `audible_tools.db`
- **Cache**: `cache/` directory for temporary audio files

## Supported Formats

**Audio**: MP3, WAV, FLAC, M4A, AAC, OGG, WMA, AIFF, AU
**Video**: MP4, MOV, AVI, MKV, WMV, FLV, WEBM (audio track extraction)

## Requirements

- Python 3.8+
- ffmpeg (for audio conversion)
- CUDA-compatible GPU (optional, for faster processing)
- Internet connection (for model downloads on first use)

## Testing

The project includes a comprehensive test suite that covers all primary features:

### Running Tests

**Basic unit tests:**
```bash
cd proof-of-concept/
source virtualenv/bin/activate
python test_audible_tools.py
```

**Verbose test output:**
```bash
python test_audible_tools.py --verbose
```

**Performance tests (uses real audio files):**
```bash
python test_audible_tools.py --performance
```

### Test Coverage

The test suite covers:
- **Database Operations**: Table creation, UUID generation, voice management
- **Audio Processing**: Format conversion, metadata extraction, transcription
- **CLI Interface**: All command-line operations and help systems
- **Integration Tests**: Full workflow using real audio files from `audio-samples/`
- **Performance Tests**: Speed benchmarks with timing measurements

### Test Files

Tests automatically use the smallest audio files from `audio-samples/` to ensure fast execution. For performance testing, medium-sized files (100KB-1MB) are preferred.

### GPU Acceleration

The tool automatically detects and uses available GPUs:
- **CUDA**: NVIDIA GPUs with CUDA support
- **MPS**: Apple Silicon Macs with Metal Performance Shaders
- **CPU**: Fallback for systems without GPU acceleration

GPU usage is displayed during model loading and significantly speeds up processing.

## GPU Support and Performance Benchmarking

### Current GPU Status

You may see the message "Using CPU (GPU support temporarily disabled due to compatibility issues)" for the following reasons:

1. **MPS Compatibility Issues**: While Apple's Metal Performance Shaders (MPS) is detected as available, it currently fails when loading the sentence transformer model due to sparse tensor operations not being fully supported.

2. **PyTorch Version Compatibility**: The current PyTorch version has incomplete MPS support for certain operations used by the sentence transformer library.

### Performance Benchmarking

To benchmark CPU vs GPU performance and understand the potential speedup, use our benchmarking tools:

#### Simple Whisper Benchmark

Test Whisper transcription performance (most compute-intensive operation):

```bash
python simple_benchmark.py
```

This will show you:
- Model loading time comparison
- Transcription time comparison  
- Real-time factor (how much faster than real-time playback)
- Speedup calculations

#### Comprehensive Benchmark

For full system benchmarking including database operations:

```bash
python benchmark.py --max-files 2
```

Options:
- `--devices cpu cuda mps` - Specify which devices to test
- `--files audio1.wma audio2.wma` - Test specific files
- `--output results.json` - Save detailed results

### Expected Performance Gains

Based on typical GPU performance improvements:

- **Apple M1/M2/M3 (MPS)**: 2-4x speedup for Whisper transcription
- **NVIDIA GPU (CUDA)**: 3-8x speedup depending on model size
- **Model Loading**: Similar times (models are small)
- **Database Operations**: No improvement (CPU-bound)

### GPU Support Troubleshooting

#### For Apple Silicon (M1/M2/M3) Users

The MPS backend fails with this error:
```
Could not run 'aten::_sparse_coo_tensor_with_dims_and_tensors' with arguments from the 'SparseMPS' backend
```

**Potential Solutions:**

1. **Update PyTorch** (may resolve in future versions):
   ```bash
   pip install --upgrade torch torchvision torchaudio
   ```

2. **Use CPU-only sentence transformer** (already implemented as fallback):
   ```bash
   # The code automatically falls back to CPU for sentence transformers
   # while using MPS for Whisper when possible
   ```

3. **Force CPU mode** if needed:
   ```bash
   export PYTORCH_ENABLE_MPS_FALLBACK=1
   ```

#### For NVIDIA GPU Users

CUDA should work without issues. Make sure you have:
- NVIDIA drivers installed
- CUDA toolkit compatible with your PyTorch version
- GPU memory available (2GB+ recommended)

### Manual GPU Testing

To test GPU compatibility manually:

```python
import torch
import whisper

# Test basic GPU operations
if torch.cuda.is_available():
    print("CUDA available")
    device = "cuda"
elif torch.backends.mps.is_available():
    print("MPS available")
    device = "mps"
else:
    device = "cpu"

# Test Whisper model loading
model = whisper.load_model("base", device=device)
print(f"Model loaded on {device}")
```

### Re-enabling GPU Support

GPU support is currently disabled by default due to compatibility issues. To re-enable it:

1. Edit `main.py` and locate the `get_device()` function
2. The function now includes proper error handling for GPU failures
3. GPU will be automatically used when available and compatible

### Performance Monitoring

When running with GPU support, you'll see:
- Device selection: "Using Apple Metal Performance Shaders (MPS)" or "Using CUDA GPU: [GPU Name]"
- Processing times in verbose mode
- Memory usage warnings if GPU memory is low

### Limitations

Current limitations with GPU support:
- **Sentence transformers**: May fall back to CPU due to sparse tensor issues
- **Memory constraints**: Large audio files may exceed GPU memory
- **Model compatibility**: Some Whisper models work better on specific devices

### Future Improvements

Planned enhancements:
- **Automatic mixed precision**: Use FP16 when available for 2x speedup
- **Batch processing**: Process multiple files simultaneously
- **Memory optimization**: Stream large audio files to prevent OOM errors
- **Model quantization**: Reduce memory usage with minimal accuracy loss

## Notes

- Models are downloaded automatically on first use
- Processing time depends on audio length and hardware
- GPU acceleration automatically detected and used when available
- Speaker diarization requires accepting pyannote.audio terms (currently disabled)
- All operations are non-destructive to original files
- Exports include timestamps for use in video players like VLC
- Tests use real audio files to ensure compatibility with the Rust implementation 