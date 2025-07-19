
# TODO: Add devices table

# Audible Tools - Audio Processing and Transcription Tool

A command-line tool for processing audio files with automatic transcription and speaker diarization.

## Features

- **Audio File Ingestion**: Supports multiple audio formats (MP3, WAV, FLAC, M4A, AAC, OGG, WMA, AIFF, AU)
- **Automatic Transcription**: Uses OpenAI Whisper for speech-to-text conversion
- **Speaker Diarization**: Identifies and separates different speakers using pyannote.audio
- **Performance Monitoring**: Built-in timing and bottleneck identification
- **Text Search**: Search through transcripts using natural language queries
- **Voice Management**: Rename and merge speaker identities
- **Export Functionality**: Generate SRT subtitle files with speaker labels
- **Database Storage**: Persistent storage using DuckDB with comprehensive schema
- **GPU Acceleration**: Automatic detection and use of CUDA/MPS/CPU

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

3. **Configure speaker diarization (REQUIRED):**
   - Visit https://huggingface.co/pyannote/speaker-diarization-3.1
   - Accept the terms and conditions
   - Get your access token from https://huggingface.co/settings/tokens
   - Create a `.env` file in the `proof-of-concept` directory:
     ```bash
     echo "HUGGINGFACE_TOKEN=your_token_here" > .env
     ```
   - Replace `your_token_here` with your actual Hugging Face token

   **Note**: The HuggingFace token is required for speaker diarization. The tool will fail to start without it.

## Usage

### Basic Commands

**Add audio files to database:**
```bash
./audible_tools.py add ./audio-file.mp3
./audible_tools.py add -R ./audio-directory/  # Recursive
```

**Check processing status:**
```bash
./audible_tools.py status
```

**List files:**
```bash
./audible_tools.py ls
./audible_tools.py ls ./audio-directory/
```

### Voice Management

**List identified voices:**
```bash
./audible_tools.py voices list
```

**Rename a voice:**
```bash
./audible_tools.py voices rename 1 "Mike"
```

### Search and Discovery

**Search transcripts:**
```bash
./audible_tools.py search "dinner plans"
./audible_tools.py search "meeting" --limit 20
```

**Find all speech by a specific voice:**
```bash
./audible_tools.py voice "Mike"
./audible_tools.py voice "Speaker_1" --limit 15
```

### Export Transcripts

**Export as SRT files:**
```bash
./audible_tools.py export ./audio-file.mp3
./audible_tools.py export -R ./audio-directory/
```

### Database Management

**Remove files from database:**
```bash
./audible_tools.py rm ./audio-file.mp3
./audible_tools.py rm -R ./audio-directory/
```

**Reset entire database:**
```bash
./audible_tools.py reset
```

## Performance Features

### Built-in Performance Monitoring

The tool includes comprehensive performance monitoring that displays timing for major operations:

```
⏱️  load_models took 3.45 seconds
⏱️  convert_audio_to_wav took 0.42 seconds
⏱️  get_audio_metadata took 0.17 seconds
⏱️  process_segments_with_diarization took 12.34 seconds
⏱️  add_files took 18.23 seconds
```

This helps identify bottlenecks and optimize hardware usage.

### GPU Acceleration

The tool automatically detects and uses available GPUs:
- **CUDA**: NVIDIA GPUs with CUDA support
- **MPS**: Apple Silicon Macs with Metal Performance Shaders
- **CPU**: Fallback for systems without GPU acceleration

GPU usage is displayed during model loading:
```
Using Apple Metal Performance Shaders (MPS)
Using CUDA GPU: NVIDIA GeForce RTX 4080
Using CPU
```

### Hardware Optimization

The implementation includes several optimizations:
- **Automatic device selection**: Chooses the best available hardware
- **Graceful fallback**: Falls back to CPU if GPU fails
- **Memory management**: Efficient cleanup of temporary files
- **Parallel processing**: Concurrent model loading where possible

## File Processing Pipeline

1. **Ingestion**: Files are added to the database with metadata extraction
2. **Conversion**: Audio is converted to WAV format for processing
3. **Transcription**: Whisper generates text transcripts with timestamps
4. **Diarization**: pyannote.audio identifies and separates speakers (optional)
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
- **Database**: `audible-tools.db`
- **Cache**: `cache/` directory for temporary audio files

## Supported Formats

**Audio**: MP3, WAV, FLAC, M4A, AAC, OGG, WMA, AIFF, AU
**Video**: MP4, MOV, AVI, MKV, WMV, FLV, WEBM (audio track extraction)

## Requirements

- Python 3.8+
- ffmpeg (for audio conversion)
- cmake (for building optional dependencies)
- CUDA-compatible GPU (optional, for faster processing)
- Internet connection (for model downloads on first use)

## Testing

The project includes a comprehensive test suite that covers all primary features using real audio files from the `audio-samples/` directory.

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

**Test speaker diarization specifically:**
```bash
python test_pyannote.py
```

### Test Coverage

The test suite covers:
- **Database Operations**: Table creation, UUID generation, voice management
- **Audio Processing**: Format conversion, metadata extraction, transcription
- **Speaker Diarization**: Segment processing with required diarization, distinct speaker creation
- **CLI Interface**: All command-line operations and help systems
- **Integration Tests**: Full workflow using real audio files from `audio-samples/`
- **Performance Tests**: Speed benchmarks with timing measurements
- **Format Support**: Tests various audio formats (WMA, WAV, etc.)
- **Pyannote Integration**: Dedicated tests for speaker diarization functionality

### Test Files

Tests use audio files from `audio-samples/` directory:
- **Unit tests**: Use smallest files for fast execution
- **Performance tests**: Use medium-sized files (100KB-1MB) for realistic benchmarks
- **Integration tests**: Test full workflows with actual audio content

### Test Results

Current test status:
- **20 tests total**: All passing
- **Performance monitoring**: Integrated into all major operations
- **Format support**: Covers multiple audio formats
- **Speaker diarization**: Required pyannote.audio integration with distinct speaker detection

## Speaker Diarization

The tool requires advanced speaker diarization using pyannote.audio:

### Configuration

**Speaker diarization is mandatory** and provides:
- Automatic identification of multiple speakers in audio files
- Speakers labeled as "Speaker SPEAKER_00", "Speaker SPEAKER_01", etc.
- Accurate speaker attribution for each transcript segment
- Support for unlimited number of speakers per audio file

### Testing Speaker Diarization

To verify that speaker diarization is working correctly:

```bash
# Test pyannote.audio functionality
python test_pyannote.py
```

This will:
- Verify pyannote.audio is properly installed
- Check that your HuggingFace token is configured
- Test the diarization pipeline loading
- Process a sample audio file to demonstrate distinct speaker identification

### Performance Impact

- **Processing time**: ~2x slower than transcription-only due to diarization overhead
- **Accuracy**: High-quality speaker separation and attribution
- **Memory usage**: Requires additional GPU/CPU memory for diarization models

## Performance Characteristics

### Typical Processing Times

Based on testing with various audio formats:
- **Model loading**: 3-5 seconds (one-time per session)
- **Audio conversion**: 0.2-0.5 seconds per file
- **Transcription**: 0.5-2x real-time (depends on hardware)
- **Diarization**: 1-3x real-time (when enabled)
- **Database operations**: <0.1 seconds per operation

### Hardware Recommendations

- **Minimum**: 4GB RAM, any CPU
- **Recommended**: 8GB RAM, dedicated GPU
- **Optimal**: 16GB RAM, NVIDIA RTX or Apple Silicon

### Expected Performance Gains

GPU acceleration provides significant speedup:
- **Apple M1/M2/M3 (MPS)**: 2-4x speedup for Whisper transcription
- **NVIDIA GPU (CUDA)**: 3-8x speedup depending on model size
- **Model Loading**: Similar times (models are small)
- **Database Operations**: No improvement (CPU-bound)

## Troubleshooting

### Common Issues

1. **Missing HuggingFace token**: Tool will fail to start without HUGGINGFACE_TOKEN configured
2. **GPU fallback to CPU**: Normal behavior when GPU has compatibility issues
3. **Slow processing**: Check GPU availability and audio file size
4. **Pipeline loading errors**: Verify network connection and HuggingFace token validity

### GPU Support Issues

For Apple Silicon Macs:
- MPS may fall back to CPU for certain operations
- This is normal and doesn't affect functionality
- Processing will still be faster than pure CPU

For NVIDIA GPUs:
- Ensure CUDA drivers are installed
- Check GPU memory (2GB+ recommended)
- Verify PyTorch CUDA compatibility

### Memory Issues

For large audio files:
- Tool automatically manages memory
- Temporary files are cleaned up
- Consider processing files in smaller batches

## Development Status

### Current Implementation

✅ **Completed**:
- Core audio processing pipeline
- Speaker diarization with pyannote.audio
- Performance monitoring and optimization
- Comprehensive test suite
- GPU acceleration with fallback
- Multiple audio format support
- Database schema and operations

### Next Steps

For Rust implementation:
- Test suite provides reference behavior
- All major operations have performance baselines
- Database schema is fully defined
- Error handling patterns established

## Notes

- Models are downloaded automatically on first use
- Processing time depends on audio length and hardware
- GPU acceleration automatically detected and used when available
- Speaker diarization requires accepting pyannote.audio terms
- All operations are non-destructive to original files
- Exports include timestamps for use in video players like VLC
- Tests use real audio files to ensure compatibility with the Rust implementation
- Performance monitoring helps identify optimization opportunities 