"""
Utility functions for audio processing and database management.
"""

# from concurrent.futures import ThreadPoolExecutor, as_completed
# from dataclasses import dataclass
# from datetime import datetime
# from functools import wraps
# from mutagen import File as MutagenFile
# from pydub import AudioSegment
# from sentence_transformers import SentenceTransformer
# from tabulate import tabulate
# import faiss
# import json
# import librosa
# import numpy as np
# import shutil
# import soundfile as sf
# import subprocess
# import sys
# import tempfile
# import time
# import warnings
# import whisper
from datetime import datetime, timezone
from dateutil import parser
from fractions import Fraction
from typing import Optional
import os
import torch

DB_PATH = "audible-tools.db"
CACHE_DIR = "cache"
# CHUNK_SIZE = 5.0  # seconds for audio embeddings
# OVERLAP_SIZE = 2.5  # seconds overlap for audio embeddings
# spell-checker: disable
# fmt: off
SUPPORTED_AUDIO_FORMATS = {".aac",".ac3",".ac4",".adts",".adx",".aea",".amr",".apm",".aptx",".argo_asf",".ast",".au",".audiotoolbox",".bit",".caf",".codec2raw",".dfpwm",".dts",".eac3",".f32be",".f32le",".f64be",".f64le",".g722",".g726",".g726le",".gsm",".iamf",".latm",".lc3",".loas",".m2a",".mlp",".mmf",".mp2",".mpa",".oga",".oma",".opus",".rso",".s16be",".s24be",".s24le",".s32be",".s32le",".sbc",".sf",".sox",".spdif",".spx",".tta",".u16be",".u24be",".u24le",".u32be",".u32le",".vidc",".voc",".w64",".wav",".wv"}
SUPPORTED_VIDEO_FORMATS = {".a64",".apng",".avif",".avs",".avs2",".avs3",".dnxhd",".evc",".fits",".gif",".h261",".h263",".h264",".hevc",".ico",".image2pipe",".ivf",".m1v",".m2v",".mjpeg",".mjpg",".mkvtimestamp_v2",".obu",".sdl",".sdl2",".vc1",".vvc",".webp"}
SUPPORTED_CONTAINER_FORMATS = {".3g2",".3gp",".aif",".aifc",".aiff",".amv",".asf",".avi",".avm2",".dv",".dvd",".f4v",".flac",".flv",".gxf",".ismv",".m2t",".m2ts",".m4a",".m4v",".mkv",".mov",".mp3",".mp4",".mpeg",".mpg",".mts",".mxf_d10",".mxf",".null",".nut",".ogg",".ogv",".ra",".rm",".roq",".smjpeg",".swf",".ts",".vcd",".vob",".webm",".wma",".wmv",".wtv"}
# fmt: on
# spell-checker: enable


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


from xml.etree import ElementTree as ET


# Change encoder to device_make
def get_device_make(probe):
    pass


def get_device_model(probe):
    pass


# get_product(),
#                 "manufacturer": get_manufacturer(probe),
# format_info.get("tags", {}).get(
#                     "encoder", format_info.get("tags", {}).get("encoded_by", None)
#                 )

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
