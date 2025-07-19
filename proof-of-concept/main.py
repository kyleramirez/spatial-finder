#!/usr/bin/env python3
"""
Audible Tools - Audio Processing and Transcription Tool
Command line interface for audio file processing, transcription, and search.
"""
from datetime import datetime, timezone
from xml.etree import ElementTree as ET
import os
from dateutil import parser


import os

# import sys
# import json
import hashlib

# import subprocess
# import tempfile
# import shutil
# import warnings
# import time
from pathlib import Path

# from datetime import datetime
from typing import List, Dict, Any, Optional  # , Tuple

# from dataclasses import dataclass
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from functools import wraps
import mimetypes
from fractions import Fraction

import click
import duckdb

# import numpy as np
import torch

# import librosa
# import soundfile as sf
import ffmpeg

# from pydub import AudioSegment
# from mutagen import File as MutagenFile
from tqdm import tqdm

# from tabulate import tabulate
# import faiss
# from sentence_transformers import SentenceTransformer
# import whisper
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Required pyannote import
# try:
#     from pyannote.audio import Pipeline
# except ImportError as e:
#     print(f"ERROR: pyannote.audio is required but not available: {e}")
#     print("Please install pyannote.audio with: pip install pyannote.audio")
#     print("And ensure you have accepted the terms at: https://huggingface.co/pyannote/speaker-diarization-3.1")
#     sys.exit(1)

# # Apply monkeypatch fix for pyannote-audio issue #1861
# # This fixes the std() degrees of freedom <= 0 error when dimension size is 1
# try:
#     import pyannote.audio.models.blocks.pooling as pooling_module

#     # Store the original torch.std function
#     original_std = torch.std

#     def patched_std(input, dim=None, correction=1, keepdim=False, out=None):
#         """
#         Patched std function that handles the degrees of freedom <= 0 error.
#         Falls back to using correction=0 when the dimension size is 1.
#         """
#         if dim is not None and correction == 1:
#             # Check if the dimension size is 1 which would cause degrees of freedom <= 0
#             if input.size(dim) <= 1:
#                 # Use unbiased estimator (correction=0) when dimension size is 1
#                 return original_std(input, dim=dim, correction=0, keepdim=keepdim, out=out)

#         # Otherwise use the original function
#         return original_std(input, dim=dim, correction=correction, keepdim=keepdim, out=out)

#     # Apply the monkeypatch to torch.std
#     torch.std = patched_std

#     # Also patch the tensor method if it exists
#     if hasattr(torch.Tensor, 'std'):
#         original_tensor_std = torch.Tensor.std

#         def patched_tensor_std(self, dim=None, correction=1, keepdim=False):
#             """Patched tensor.std method that handles degrees of freedom <= 0."""
#             if dim is not None and correction == 1:
#                 if self.size(dim) <= 1:
#                     return original_tensor_std(self, dim=dim, correction=0, keepdim=keepdim)
#             return original_tensor_std(self, dim=dim, correction=correction, keepdim=keepdim)

#         torch.Tensor.std = patched_tensor_std

#     print("Applied monkeypatch fix for pyannote-audio std() degrees of freedom issue")

# except Exception as e:
#     print(f"Warning: Could not apply pyannote-audio monkeypatch: {e}")
#     print("Continuing without monkeypatch - you may encounter std() warnings")

# # Suppress specific warnings
# warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
# warnings.filterwarnings("ignore", category=UserWarning, module="whisper")

# Configuration
DB_PATH = "audible_tools.db"
CACHE_DIR = "cache"

# spell-checker: disable
# fmt: off
# TODO: DO THIS ALL OVER AGAIN BECAUSE IT WAS MISSING WMA
SUPPORTED_AUDIO_FORMATS = {".aac", ".ac3", ".ac4", ".adx", ".aea", ".amr", ".apm", ".aptx", ".argo_asf", ".ast", ".au", ".audiotoolbox", ".bit", ".caf", ".codec2raw", ".dfpwm", ".dts", ".eac3", ".f32be", ".f32le", ".f64be", ".f64le", ".g722", ".g726", ".g726le", ".gsm", ".iamf", ".latm", ".lc3", ".mlp", ".mmf", ".mp2", ".oga", ".oma", ".opus", ".rso", ".s16be", ".s24be", ".s24le", ".s32be", ".s32le", ".sbc", ".sf", ".sox", ".spdif", ".spx", ".tta", ".u16be", ".u24be", ".u24le", ".u32be", ".u32le", ".vidc", ".voc", ".w64", ".wav", ".wv", ".wma"}
SUPPORTED_VIDEO_FORMATS = {".a64",".apng",".avif",".avs",".avs3",".dnxhd",".evc",".fits",".gif",".h261",".h263",".h264",".hevc",".ico",".image2pipe",".ivf",".mjpg",".mkvtimestamp_v2",".obu",".sdl",".sdl2",".vc1",".vvc",".webp"}
SUPPORTED_CONTAINER_FORMATS = {".3g2",".3gp",".amv",".asf",".avi",".avm2",".dv",".dvd",".f4v",".flv",".gxf",".ismv",".mov",".mp4",".mpg",".mxf_d10",".mxf",".nut",".ogg",".ogv",".rm",".roq",".smjpeg",".swf",".vcd",".vob",".webm",".wtv"}
# fmt: on
# spell-checker: enable

# CHUNK_SIZE = 5.0  # seconds for audio embeddings
# OVERLAP_SIZE = 2.5  # seconds overlap for audio embeddings


# # Performance monitoring
# def performance_monitor(func):
#     """Decorator to monitor function execution time."""
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         duration = end_time - start_time
#         print(f"⏱️  {func.__name__} took {duration:.2f} seconds")
#         return result
#     return wrapper
def normalize_to_utc(dt_str: str) -> datetime:
    dt = parser.isoparse(dt_str)
    return dt.astimezone(timezone.utc)


def extract_creation_time(ffprobe_result: dict, file_path: str) -> Optional[datetime]:
    # 1. format.tags.creation_time
    creation_time = ffprobe_result.get("format", {}).get("tags", {}).get("creation_time")
    if creation_time:
        date = ffprobe_result.get("format", {}).get("tags", {}).get("date")
        if date:
            return normalize_to_utc(date + "T" + creation_time)
        return normalize_to_utc(creation_time)
    # 2. any stream.tags.creation_time
    for stream in ffprobe_result.get("streams", []):
        creation_time = stream.get("tags", {}).get("creation_time")
        if creation_time:
            return normalize_to_utc(creation_time)
    # 3. Fallback: File system timestamps
    try:
        stat = os.stat(file_path)
        ts = getattr(stat, "st_birthtime", None)  # macOS / BSD
        if not ts:
            ts = stat.st_mtime  # Fallback: last modified time
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        return None


def get_frame_rate(video_stream):
    try:
        rate_str = video_stream.get("r_frame_rate", "0/1")
        rate = float(Fraction(rate_str))
        return "{:.2f}".format(rate)
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def get_device(verbose=True, require_gpu=False):
    """Returns the best available torch.device: CUDA > MPS > CPU."""
    # Try CUDA (NVIDIA)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            torch.tensor([1.0], device="cuda")  # Sanity check
            device = torch.device("cuda")
            if verbose:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"Using CUDA GPU: {gpu_name}")
            return device
        except Exception as e:
            if verbose:
                print(f"CUDA is available but not usable: {e}. Trying MPS...")

    # Try MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        try:
            test_tensor = torch.tensor([1.0], device="mps")
            _ = test_tensor * 2  # Sanity check
            device = torch.device("mps")
            if verbose:
                print("Using Apple Metal Performance Shaders (MPS)")
            return device
        except Exception as e:
            if verbose:
                print(f"MPS is available but not usable: {e}. Falling back to CPU...")

    if require_gpu:
        raise RuntimeError("No usable GPU device found.")

    # Fallback to CPU
    if verbose:
        print("Using CPU")
    return torch.device("cpu")


class AudioDatabase:
    """Database management for audio files and processing results."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self.init_database()

    def init_database(self):
        """Initialize database schema."""
        # Create sequences for auto-incrementing primary keys
        sequences = [
            "CREATE SEQUENCE IF NOT EXISTS voices_id_seq",
            "CREATE SEQUENCE IF NOT EXISTS verbalizations_id_seq",
            "CREATE SEQUENCE IF NOT EXISTS nonverbal_labels_id_seq",
            "CREATE SEQUENCE IF NOT EXISTS silents_id_seq",
            "CREATE SEQUENCE IF NOT EXISTS audible_embeddings_id_seq",
            "CREATE SEQUENCE IF NOT EXISTS exports_id_seq",
            "CREATE SEQUENCE IF NOT EXISTS audible_faiss_indexes_id_seq",
        ]

        for seq_sql in sequences:
            self.conn.execute(seq_sql)

        # Create all tables according to the schema
        tables = [
            """
            CREATE TABLE IF NOT EXISTS audible_files (
                uuid TEXT PRIMARY KEY,
                strategy TEXT DEFAULT 'local_disk',
                path TEXT NOT NULL,
                basename TEXT NOT NULL,
                extension TEXT NOT NULL,
                mime TEXT,
                sample_rate INTEGER,
                bitrate INTEGER,
                bit_depth TEXT,
                channels TEXT,
                codec TEXT,
                codec_long TEXT,
                has_video BOOLEAN DEFAULT false,
                video_codec TEXT,
                video_bitrate INTEGER,
                video_bit_depth TEXT,
                color_space TEXT,
                profile TEXT,
                video_codec_long TEXT,
                resolution TEXT,
                frame_rate REAL,
                container_format TEXT,
                container_format_long TEXT,
                creation_date TIMESTAMP,
                generalized_location TEXT,
                latitude REAL,
                longitude REAL,
                duration REAL,
                ingest_status TEXT DEFAULT 'QUEUED',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS voices (
                id INTEGER PRIMARY KEY DEFAULT nextval('voices_id_seq'),
                display_name TEXT DEFAULT 'Unknown',
                best_verbalization_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS verbalizations (
                id INTEGER PRIMARY KEY DEFAULT nextval('verbalizations_id_seq'),
                voice_id INTEGER NOT NULL,
                audible_file_uuid TEXT NOT NULL,
                start_byte INTEGER,
                end_byte INTEGER,
                start_time REAL,
                duration REAL,
                label TEXT,
                label_confidence REAL,
                label_embedding BLOB,
                label_model TEXT,
                voice_confidence REAL,
                voice_embedding BLOB,
                voice_model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (voice_id) REFERENCES voices(id),
                FOREIGN KEY (audible_file_uuid) REFERENCES audible_files(uuid)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS nonverbal_labels (
                id INTEGER PRIMARY KEY DEFAULT nextval('nonverbal_labels_id_seq'),
                audible_file_uuid TEXT,
                start_byte INTEGER,
                end_byte INTEGER,
                start_time REAL,
                duration REAL,
                label TEXT,
                source TEXT,
                kind TEXT,
                confidence REAL,
                embedding BLOB,
                embedding_model TEXT,
                include_in_export BOOLEAN DEFAULT false,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (audible_file_uuid) REFERENCES audible_files(uuid)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS silents (
                id INTEGER PRIMARY KEY DEFAULT nextval('silents_id_seq'),
                audible_file_uuid TEXT,
                start_byte INTEGER,
                end_byte INTEGER,
                start_time REAL,
                duration REAL,
                confidence REAL,
                detector TEXT,
                detector_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (audible_file_uuid) REFERENCES audible_files(uuid)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS audible_embeddings (
                id INTEGER PRIMARY KEY DEFAULT nextval('audible_embeddings_id_seq'),
                audible_file_uuid TEXT,
                start_byte INTEGER,
                end_byte INTEGER,
                start_time REAL,
                duration REAL,
                embedding BLOB,
                embedding_model TEXT,
                segment_index INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (audible_file_uuid) REFERENCES audible_files(uuid)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS exports (
                id INTEGER PRIMARY KEY DEFAULT nextval('exports_id_seq'),
                status TEXT DEFAULT 'QUEUED',
                strategy TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS export_voices (
                voice_id INTEGER,
                display_name TEXT,
                export_id INTEGER,
                PRIMARY KEY (voice_id, export_id),
                FOREIGN KEY (voice_id) REFERENCES voices(id),
                FOREIGN KEY (export_id) REFERENCES exports(id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS audible_faiss_indexes (
                id INTEGER PRIMARY KEY DEFAULT nextval('audible_faiss_indexes_id_seq'),
                type TEXT,
                strategy TEXT,
                index_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
        ]

        for table_sql in tables:
            self.conn.execute(table_sql)

    def get_file_uuid(self, file_path: str) -> str:
        """Generate a deterministic UUID based on file contents."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    # def get_or_create_default_voice(self) -> int:
    #     """Get or create the default voice for single-speaker files."""
    #     existing = self.conn.execute(
    #         "SELECT id FROM voices WHERE display_name = 'Default Speaker'"
    #     ).fetchone()

    #     if existing:
    #         return existing[0]
    #     else:
    #         voice_id = self.conn.execute(
    #             "INSERT INTO voices (display_name) VALUES ('Default Speaker') RETURNING id"
    #         ).fetchone()[0]
    #         return voice_id


#     def close(self):
#         """Close database connection."""
#         self.conn.close()


class AudioProcessor:
    """Audio processing functionality."""

    def __init__(self, db: AudioDatabase):
        self.db = db
        self.whisper_model = None
        self.sentence_transformer = None
        self.diarization_pipeline = None
        self.device = get_device()
        self.cache_dir = Path(CACHE_DIR)
        self.cache_dir.mkdir(exist_ok=True)

    # @performance_monitor
    def load_models(self):
        """Load ML models on demand."""
        if self.whisper_model is None:
            print("Loading Whisper model...")
            # Try the selected device first, but fall back to CPU if it fails
            # whisper_device = self.device

            # with warnings.catch_warnings():
            #     warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
            #     warnings.filterwarnings("ignore", category=UserWarning, module="whisper")

            #     try:
            #         self.whisper_model = whisper.load_model("base", device=whisper_device)
            #     except Exception as e:
            #         if whisper_device == "mps":
            #             print(f"MPS failed for Whisper model ({str(e)[:100]}...), using CPU")
            #             whisper_device = "cpu"
            #             self.whisper_model = whisper.load_model("base", device=whisper_device)
            #             # Update the device for all future operations
            #             self.device = "cpu"
            #         else:
            #             raise e

        if self.sentence_transformer is None:
            print("Loading sentence transformer...")
            # Try MPS first if available, but fall back to CPU if it fails
            # sentence_device = self.device
            # try:
            #     self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2', device=sentence_device)
            # except Exception as e:
            #     if sentence_device == "mps":
            #         print(f"MPS failed for sentence transformer ({str(e)[:100]}...), using CPU")
            #         sentence_device = "cpu"
            #         self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2', device=sentence_device)
            #     else:
            #         raise e

        if self.diarization_pipeline is None:
            print("Loading speaker diarization pipeline...")
            # hf_token = os.getenv('HUGGINGFACE_TOKEN')
            # if not hf_token:
            #     print("ERROR: HUGGINGFACE_TOKEN is required for speaker diarization.")
            #     print("Please:")
            #     print("1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1")
            #     print("2. Accept the terms and conditions")
            #     print("3. Get your token from https://huggingface.co/settings/tokens")
            #     print("4. Create a .env file with: HUGGINGFACE_TOKEN=your_token_here")
            #     raise RuntimeError("HUGGINGFACE_TOKEN is required for speaker diarization")

            # try:
            #     self.diarization_pipeline = Pipeline.from_pretrained(
            #         "pyannote/speaker-diarization-3.1",
            #         use_auth_token=hf_token
            #     )

            #     # Try to move to appropriate device
            #     if self.device != "cpu":
            #         try:
            #             self.diarization_pipeline.to(torch.device(self.device))
            #         except Exception as e:
            #             print(f"Failed to move diarization pipeline to {self.device}: {e}")
            #             print("Continuing with CPU for diarization...")

            # except Exception as e:
            #     print(f"ERROR: Failed to load speaker diarization pipeline: {e}")
            #     print("This is a required feature. Please check your HUGGINGFACE_TOKEN and network connection.")
            #     raise RuntimeError(f"Failed to load speaker diarization pipeline: {e}")

    #     @performance_monitor
    def get_audio_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract audio metadata using ffprobe."""
        try:
            probe = ffmpeg.probe(file_path)
            format_info = probe["format"]

            audio_stream = None
            video_stream = None

            for stream in probe["streams"]:
                if stream["codec_type"] == "audio" and audio_stream is None:
                    audio_stream = stream
                elif stream["codec_type"] == "video" and video_stream is None:
                    video_stream = stream
            creation_date = extract_creation_time(probe, file_path)

            metadata = {
                "mime": mimetypes.guess_type(file_path)[0] or "application/octet-stream",
                "duration": float(format_info.get("duration", 0)),
                "bitrate": int(format_info.get("bit_rate", 0)),
                "container_format": format_info.get("format_name", "").split(",")[0] or None,
                "container_format_long": format_info.get("format_long_name", "").split(",")[0] or None,
                "has_video": video_stream is not None,
                "creation_date": creation_date,
            }
            if audio_stream:
                metadata.update(
                    {
                        "sample_rate": int(audio_stream.get("sample_rate", None)),
                        "channels": str(audio_stream.get("channels", None)),
                        "codec": audio_stream.get("codec_name", None),
                        "codec_long": audio_stream.get("codec_long_name", None),
                        "bit_depth": audio_stream.get("bits_per_sample", None),
                    }
                )

            if video_stream:
                width = video_stream.get("width")
                height = video_stream.get("height")
                metadata.update(
                    {
                        "video_codec": video_stream.get("codec_name", None),
                        "video_codec_long": video_stream.get("codec_long_name", None),
                        "resolution": f"{width}x{height}" if width and height else None,
                        "frame_rate": get_frame_rate(video_stream),
                        "video_bitrate": video_stream.get("bit_rate", None),
                        "video_bit_depth": video_stream.get("bits_per_raw_sample", None),
                        "color_space": video_stream.get("color_space", None),
                        "profile": video_stream.get("profile", None),
                    }
                )

            return metadata

        except Exception as e:
            print(f"Error getting metadata for {file_path}: {e}")
            return {}

    #     @performance_monitor
    #     def convert_audio_to_wav(self, input_path: str, output_path: str) -> bool:
    #         """Convert audio file to WAV format."""
    #         try:
    #             # Use ffmpeg for conversion
    #             (
    #                 ffmpeg
    #                 .input(input_path)
    #                 .output(output_path, acodec='pcm_s16le', ac=1, ar=16000)
    #                 .overwrite_output()
    #                 .run(quiet=True)
    #             )
    #             return True
    #         except Exception as e:
    #             print(f"Error converting {input_path} to WAV: {e}")
    #             return False

    #     @performance_monitor
    def process_audio_file(self, file_path: str, file_uuid: str) -> bool:
        """Process a single audio file with Whisper and speaker diarization."""
        try:
            self.load_models()

            # Convert to WAV if needed
        #     wav_path = self.cache_dir / f"{file_uuid}.wav"
        #     if not self.convert_audio_to_wav(file_path, str(wav_path)):
        #         return False

        #     # Transcribe with Whisper
        #     print(f"Transcribing {Path(file_path).name}...")
        #     # Suppress FP16 warnings during transcription
        #     with warnings.catch_warnings():
        #         warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
        #         warnings.filterwarnings("ignore", category=UserWarning, module="whisper")
        #         result = self.whisper_model.transcribe(str(wav_path))

        #     # Perform speaker diarization (required)
        #     print(f"Performing speaker diarization for {Path(file_path).name}...")
        #     try:
        #         diarization_result = self.diarization_pipeline(str(wav_path))
        #     except Exception as e:
        #         print(f"ERROR: Speaker diarization failed: {e}")
        #         raise RuntimeError(f"Speaker diarization failed: {e}")

        #     # Process segments with speaker diarization
        #     success = self.process_segments_with_diarization(file_uuid, result, diarization_result)

        #     if success:
        #         # Update status
        #         self.db.conn.execute(
        #             "UPDATE audible_files SET ingest_status = 'COMPLETE', updated_at = CURRENT_TIMESTAMP WHERE uuid = ?",
        #             (file_uuid,)
        #         )
        #     else:
        #         # Update status to failed
        #         self.db.conn.execute(
        #             "UPDATE audible_files SET ingest_status = 'FAILED', updated_at = CURRENT_TIMESTAMP WHERE uuid = ?",
        #             (file_uuid,)
        #         )

        #     # Clean up temp file
        #     if wav_path.exists():
        #         wav_path.unlink()

        #     return success

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            self.db.conn.execute(
                "UPDATE audible_files SET ingest_status = 'FAILED', updated_at = CURRENT_TIMESTAMP WHERE uuid = ?",
                (file_uuid,),
            )
            return False


#     @performance_monitor
#     def process_segments_with_diarization(self, file_uuid: str, whisper_result: Dict, diarization_result) -> bool:
#         """Process Whisper segments with speaker diarization."""
#         try:
#             # Create speaker mapping
#             speaker_map = {}

#             # Create voices for each speaker found in diarization
#             for turn, _, speaker in diarization_result.itertracks(yield_label=True):
#                 if speaker not in speaker_map:
#                     # Create new voice for this speaker
#                     voice_id = self.db.conn.execute(
#                         "INSERT INTO voices (display_name) VALUES (?) RETURNING id",
#                         (f"Speaker {speaker}",)
#                     ).fetchone()[0]
#                     speaker_map[speaker] = voice_id

#             # Process each segment
#             for segment in whisper_result['segments']:
#                 start_time = segment['start']
#                 end_time = segment['end']
#                 text = segment['text'].strip()

#                 # Skip empty segments
#                 if not text:
#                     continue

#                 confidence = segment.get('avg_logprob', 0.0)

#                 # Find the speaker for this segment
#                 voice_id = None
#                 voice_confidence = 0.0

#                 # Find overlapping speaker segment
#                 segment_midpoint = (start_time + end_time) / 2
#                 best_overlap = 0
#                 best_speaker = None

#                 for turn, _, speaker in diarization_result.itertracks(yield_label=True):
#                     overlap_start = max(start_time, turn.start)
#                     overlap_end = min(end_time, turn.end)
#                     overlap_duration = max(0, overlap_end - overlap_start)

#                     if overlap_duration > best_overlap:
#                         best_overlap = overlap_duration
#                         best_speaker = speaker

#                 if best_speaker:
#                     voice_id = speaker_map[best_speaker]
#                     voice_confidence = best_overlap / (end_time - start_time)
#                 else:
#                     # If no speaker overlap found, skip this segment or assign to first speaker
#                     if speaker_map:
#                         voice_id = list(speaker_map.values())[0]
#                         voice_confidence = 0.0
#                     else:
#                         print(f"Warning: No speakers found for segment at {start_time:.2f}s")
#                         continue

#                 # Generate text embedding
#                 text_embedding = self.sentence_transformer.encode(text)
#                 text_embedding_bytes = text_embedding.astype(np.float32).tobytes()

#                 # Insert verbalization
#                 self.db.conn.execute(
#                     """
#                     INSERT INTO verbalizations
#                     (voice_id, audible_file_uuid, start_time, duration, label, label_confidence,
#                      label_embedding, label_model, voice_confidence, voice_model)
#                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#                     """,
#                     (
#                         voice_id, file_uuid, start_time, end_time - start_time,
#                         text, confidence, text_embedding_bytes, "all-MiniLM-L6-v2",
#                         voice_confidence, "pyannote-speaker-diarization-3.1"
#                     )
#                 )

#             return True

#         except Exception as e:
#             print(f"Error processing segments: {e}")
#             return False


class AudioToolsCLI:
    """Command line interface for audio tools."""

    def __init__(self):
        self.db = AudioDatabase()
        self.processor = AudioProcessor(self.db)

    #     @performance_monitor
    def add_files(self, paths: List[str], recursive: bool = False):
        """Add audio files to the database."""
        files_to_process = []

        for path_str in paths:
            path = Path(path_str)
            if path.is_file():
                if (
                    path.suffix.lower() in SUPPORTED_AUDIO_FORMATS
                    or path.suffix.lower() in SUPPORTED_VIDEO_FORMATS
                    or path.suffix.lower() in SUPPORTED_CONTAINER_FORMATS
                ):
                    files_to_process.append(path)
            elif path.is_dir() and recursive:
                for ext in SUPPORTED_AUDIO_FORMATS | SUPPORTED_VIDEO_FORMATS | SUPPORTED_CONTAINER_FORMATS:
                    files_to_process.extend(path.rglob(f"*{ext}"))

        if not files_to_process:
            print("No audio files found to process.")
            return

        print(f"Found {len(files_to_process)} files to process.")

        # Process files
        for file_path in tqdm(files_to_process, desc="Adding files"):
            file_uuid = self.db.get_file_uuid(str(file_path))
            # Check if already exists
            existing = self.db.conn.execute("SELECT uuid FROM audible_files WHERE uuid = ?", (file_uuid,)).fetchone()
            if existing:
                print(f"Skipping {file_path.name} (already exists)")
                continue

            # Get metadata
            metadata = self.processor.get_audio_metadata(str(file_path))

            # Insert file record
            self.db.conn.execute(
                """
                INSERT INTO audible_files 
                (
                    uuid,
                    path,
                    basename,
                    extension,
                    mime,
                    sample_rate,
                    bitrate,
                    bit_depth,
                    channels,
                    codec,
                    codec_long,
                    has_video,
                    video_codec,
                    video_codec_long,
                    resolution,
                    frame_rate,
                    container_format,
                    container_format_long,
                    duration,
                    creation_date,
                    video_bitrate,
                    video_bit_depth,
                    color_space,
                    profile
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_uuid,
                    str(file_path),
                    file_path.name,
                    file_path.suffix,
                    metadata.get("mime"),
                    metadata.get("sample_rate"),
                    metadata.get("bitrate"),
                    metadata.get("bit_depth"),
                    metadata.get("channels"),
                    metadata.get("codec"),
                    metadata.get("codec_long"),
                    metadata.get("has_video", False),
                    metadata.get("video_codec"),
                    metadata.get("video_codec_long"),
                    metadata.get("resolution"),
                    metadata.get("frame_rate"),
                    metadata.get("container_format"),
                    metadata.get("container_format_long"),
                    metadata.get("duration"),
                    metadata.get("creation_date"),
                    metadata.get("video_bitrate"),
                    metadata.get("video_bit_depth"),
                    metadata.get("color_space"),
                    metadata.get("profile"),
                ),
            )

        # Process queued files
        self.process_queued_files()

    #     @performance_monitor
    def process_queued_files(self):
        """Process all queued files."""
        queued_files = self.db.conn.execute(
            "SELECT uuid, path FROM audible_files WHERE ingest_status = 'QUEUED'"
        ).fetchall()

        if not queued_files:
            print("No files to process.")
            return

        print(f"Processing {len(queued_files)} queued files...")

        for file_uuid, file_path in tqdm(queued_files, desc="Processing"):
            # Update status to working
            self.db.conn.execute(
                "UPDATE audible_files SET ingest_status = 'WORKING', updated_at = CURRENT_TIMESTAMP WHERE uuid = ?",
                (file_uuid,),
            )

            # Process the file
            self.processor.process_audio_file(file_path, file_uuid)


#     def show_status(self):
#         """Show processing status."""
#         stats = self.db.conn.execute(
#             """
#             SELECT
#                 ingest_status,
#                 COUNT(*) as count,
#                 COALESCE(SUM(duration), 0) as total_duration
#             FROM audible_files
#             GROUP BY ingest_status
#             """
#         ).fetchall()

#         print("\nProcessing Status:")
#         headers = ["Status", "Count", "Total Duration (min)"]
#         rows = [(status, count, round(duration/60, 2)) for status, count, duration in stats]
#         print(tabulate(rows, headers=headers, tablefmt="grid"))

#         # Voice stats
#         voice_count = self.db.conn.execute("SELECT COUNT(*) FROM voices").fetchone()[0]
#         verbalization_count = self.db.conn.execute("SELECT COUNT(*) FROM verbalizations").fetchone()[0]

#         print(f"\nVoices identified: {voice_count}")
#         print(f"Verbalizations processed: {verbalization_count}")

#     def list_files(self, path: str = None):
#         """List files with their processing status."""
#         if path:
#             query = """
#             SELECT basename, ingest_status, duration,
#                    (SELECT COUNT(*) FROM verbalizations WHERE audible_file_uuid = af.uuid) as transcripts,
#                    (SELECT COUNT(DISTINCT voice_id) FROM verbalizations WHERE audible_file_uuid = af.uuid) as voices
#             FROM audible_files af
#             WHERE path LIKE ?
#             ORDER BY basename
#             """
#             results = self.db.conn.execute(query, (f"%{path}%",)).fetchall()
#         else:
#             query = """
#             SELECT basename, ingest_status, duration,
#                    (SELECT COUNT(*) FROM verbalizations WHERE audible_file_uuid = af.uuid) as transcripts,
#                    (SELECT COUNT(DISTINCT voice_id) FROM verbalizations WHERE audible_file_uuid = af.uuid) as voices
#             FROM audible_files af
#             ORDER BY basename
#             """
#             results = self.db.conn.execute(query).fetchall()

#         headers = ["File", "Status", "Duration (min)", "Transcripts", "Voices"]
#         rows = [(name, status, round(duration/60, 2) if duration else 0, transcripts, voices)
#                 for name, status, duration, transcripts, voices in results]

#         print(tabulate(rows, headers=headers, tablefmt="grid"))

#     def list_voices(self):
#         """List all identified voices."""
#         voices = self.db.conn.execute(
#             """
#             SELECT v.id, v.display_name, COUNT(vb.id) as verbalizations
#             FROM voices v
#             LEFT JOIN verbalizations vb ON v.id = vb.voice_id
#             GROUP BY v.id, v.display_name
#             ORDER BY verbalizations DESC
#             """
#         ).fetchall()

#         headers = ["ID", "Name", "Verbalizations"]
#         print(tabulate(voices, headers=headers, tablefmt="grid"))

#     def rename_voice(self, voice_id: int, new_name: str):
#         """Rename a voice."""
#         self.db.conn.execute(
#             "UPDATE voices SET display_name = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
#             (new_name, voice_id)
#         )
#         print(f"Voice {voice_id} renamed to '{new_name}'")

#     def search_text(self, query: str, limit: int = 10):
#         """Search for text in verbalizations."""
#         results = self.db.conn.execute(
#             """
#             SELECT v.display_name, vb.label, vb.start_time, af.basename
#             FROM verbalizations vb
#             JOIN voices v ON vb.voice_id = v.id
#             JOIN audible_files af ON vb.audible_file_uuid = af.uuid
#             WHERE vb.label LIKE ?
#             ORDER BY vb.label_confidence DESC
#             LIMIT ?
#             """,
#             (f"%{query}%", limit)
#         ).fetchall()

#         headers = ["Speaker", "Text", "Time (min)", "File"]
#         rows = [(name, text[:100] + "..." if len(text) > 100 else text, round(time/60, 2), file)
#                 for name, text, time, file in results]

#         print(f"\nSearch results for '{query}':")
#         print(tabulate(rows, headers=headers, tablefmt="grid"))

#     def search_voice(self, voice_name: str, limit: int = 10):
#         """Search for verbalizations by voice."""
#         results = self.db.conn.execute(
#             """
#             SELECT vb.label, vb.start_time, af.basename
#             FROM verbalizations vb
#             JOIN voices v ON vb.voice_id = v.id
#             JOIN audible_files af ON vb.audible_file_uuid = af.uuid
#             WHERE v.display_name LIKE ?
#             ORDER BY vb.start_time
#             LIMIT ?
#             """,
#             (f"%{voice_name}%", limit)
#         ).fetchall()

#         headers = ["Text", "Time (min)", "File"]
#         rows = [(text[:100] + "..." if len(text) > 100 else text, round(time/60, 2), file)
#                 for text, time, file in results]

#         print(f"\nVerbalizations by '{voice_name}':")
#         print(tabulate(rows, headers=headers, tablefmt="grid"))

#     def export_transcripts(self, paths: List[str], recursive: bool = False):
#         """Export transcripts as sidecar files."""
#         files_to_export = []

#         for path_str in paths:
#             path = Path(path_str)
#             if path.is_file():
#                 files_to_export.append(path)
#             elif path.is_dir() and recursive:
#                 # Find all processed audio files in directory
#                 processed_files = self.db.conn.execute(
#                     "SELECT path FROM audible_files WHERE ingest_status = 'COMPLETE' AND path LIKE ?",
#                     (f"%{path}%",)
#                 ).fetchall()
#                 files_to_export.extend([Path(p[0]) for p in processed_files])

#         if not files_to_export:
#             print("No files found to export.")
#             return

#         for file_path in tqdm(files_to_export, desc="Exporting transcripts"):
#             self.export_single_file(file_path)

#     def export_single_file(self, file_path: Path):
#         """Export transcript for a single file."""
#         file_uuid = self.db.get_file_uuid(str(file_path))

#         # Get verbalizations
#         verbalizations = self.db.conn.execute(
#             """
#             SELECT vb.start_time, vb.duration, vb.label, v.display_name
#             FROM verbalizations vb
#             JOIN voices v ON vb.voice_id = v.id
#             WHERE vb.audible_file_uuid = ?
#             ORDER BY vb.start_time
#             """,
#             (file_uuid,)
#         ).fetchall()

#         if not verbalizations:
#             print(f"No transcript found for {file_path.name}")
#             return

#         # Generate SRT content
#         srt_content = []
#         for i, (start_time, duration, text, speaker) in enumerate(verbalizations, 1):
#             start_srt = self.seconds_to_srt_time(start_time)
#             end_srt = self.seconds_to_srt_time(start_time + duration)

#             srt_content.append(f"{i}")
#             srt_content.append(f"{start_srt} --> {end_srt}")
#             srt_content.append(f"{speaker}: {text}")
#             srt_content.append("")

#         # Write SRT file
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         srt_path = file_path.with_suffix(f".exported_{timestamp}.srt")

#         with open(srt_path, 'w', encoding='utf-8') as f:
#             f.write('\n'.join(srt_content))

#         print(f"Exported transcript: {srt_path}")

#     def seconds_to_srt_time(self, seconds: float) -> str:
#         """Convert seconds to SRT time format."""
#         hours = int(seconds // 3600)
#         minutes = int((seconds % 3600) // 60)
#         secs = int(seconds % 60)
#         millis = int((seconds % 1) * 1000)
#         return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

#     def remove_files(self, paths: List[str], recursive: bool = False):
#         """Remove files from database."""
#         files_to_remove = []

#         for path_str in paths:
#             path = Path(path_str)
#             if path.is_file():
#                 files_to_remove.append(path)
#             elif path.is_dir() and recursive:
#                 # Find all files in directory
#                 db_files = self.db.conn.execute(
#                     "SELECT path FROM audible_files WHERE path LIKE ?",
#                     (f"%{path}%",)
#                 ).fetchall()
#                 files_to_remove.extend([Path(p[0]) for p in db_files])

#         for file_path in files_to_remove:
#             file_uuid = self.db.get_file_uuid(str(file_path))

#             # Remove from database
#             self.db.conn.execute("DELETE FROM verbalizations WHERE audible_file_uuid = ?", (file_uuid,))
#             self.db.conn.execute("DELETE FROM audible_files WHERE uuid = ?", (file_uuid,))

#             print(f"Removed {file_path.name} from database")

#     def reset_database(self):
#         """Reset the entire database."""
#         confirm = input("This will delete all data. Are you sure? (yes/no): ")
#         if confirm.lower() == 'yes':
#             self.db.conn.execute("DROP TABLE IF EXISTS verbalizations")
#             self.db.conn.execute("DROP TABLE IF EXISTS voices")
#             self.db.conn.execute("DROP TABLE IF EXISTS audible_files")
#             self.db.conn.execute("DROP TABLE IF EXISTS nonverbal_labels")
#             self.db.conn.execute("DROP TABLE IF EXISTS silents")
#             self.db.conn.execute("DROP TABLE IF EXISTS audible_embeddings")
#             self.db.conn.execute("DROP TABLE IF EXISTS exports")
#             self.db.conn.execute("DROP TABLE IF EXISTS export_voices")
#             self.db.conn.execute("DROP TABLE IF EXISTS audible_faiss_indexes")

#             self.db.init_database()
#             print("Database reset complete.")
#         else:
#             print("Reset cancelled.")


# CLI Command definitions
@click.group()
def cli():
    """Audible Tools - Audio Processing and Transcription Tool"""
    pass


@cli.command()
@click.argument("paths", nargs=-1, required=True)
@click.option("-R", "--recursive", is_flag=True, help="Process directories recursively")
def add(paths, recursive):
    """Add audio files to the database."""
    tool = AudioToolsCLI()
    tool.add_files(list(paths), recursive)


# @cli.command()
# def status():
#     """Show processing status."""
#     tool = AudioToolsCLI()
#     tool.show_status()

# @cli.command()
# @click.argument('path', required=False)
# def ls(path):
#     """List files with their processing status."""
#     tool = AudioToolsCLI()
#     tool.list_files(path)

# @cli.group()
# def voices():
#     """Voice management commands."""
#     pass

# @voices.command('list')
# def voices_list():
#     """List all identified voices."""
#     tool = AudioToolsCLI()
#     tool.list_voices()

# @voices.command('rename')
# @click.argument('voice_id', type=int)
# @click.argument('new_name')
# def voices_rename(voice_id, new_name):
#     """Rename a voice."""
#     tool = AudioToolsCLI()
#     tool.rename_voice(voice_id, new_name)

# @cli.command()
# @click.argument('query')
# @click.option('--limit', default=10, help='Maximum number of results')
# def search(query, limit):
#     """Search for text in verbalizations."""
#     tool = AudioToolsCLI()
#     tool.search_text(query, limit)

# @cli.command()
# @click.argument('voice_name')
# @click.option('--limit', default=10, help='Maximum number of results')
# def voice(voice_name, limit):
#     """Search for verbalizations by voice."""
#     tool = AudioToolsCLI()
#     tool.search_voice(voice_name, limit)

# @cli.command()
# @click.argument('paths', nargs=-1, required=True)
# @click.option('-R', '--recursive', is_flag=True, help='Process directories recursively')
# def export(paths, recursive):
#     """Export transcripts as sidecar files."""
#     tool = AudioToolsCLI()
#     tool.export_transcripts(list(paths), recursive)

# @cli.command()
# @click.argument('paths', nargs=-1, required=True)
# @click.option('-R', '--recursive', is_flag=True, help='Process directories recursively')
# def rm(paths, recursive):
#     """Remove files from database."""
#     tool = AudioToolsCLI()
#     tool.remove_files(list(paths), recursive)

# @cli.command()
# def reset():
#     """Reset the entire database."""
#     tool = AudioToolsCLI()
#     tool.reset_database()

if __name__ == "__main__":
    cli()
