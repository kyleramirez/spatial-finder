from pathlib import Path
from typing import Dict, Any
import ffmpeg
import mimetypes
from .utils import CACHE_DIR, get_device, extract_creation_time, get_frame_rate, get_device_make, get_device_model
from .audio_database import AudioDatabase


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
                "container_format": format_info.get("format_long_name", None),
                "has_video": video_stream is not None,
                "creation_date": creation_date,
                "device_make": get_device_make(probe),
                "device_model": get_device_model(probe),
            }
            # Rename encoder column to device_make
            # Add device_model TEXT column
            # Remove container_format_long, codec_long, video_codec_long columns
            # Check panasonic video codec and put that in place of codec_long_name in a utility function
            # Write more utility functions to extract metadata
            # have a place to put H264_422_LongGOP
            # use the panasonic XML frame rate
            # Find out if it's interlaced or progressive
            # Record timecode if available
            # Use the Panasonic CINELIKE_D and BT.709 values
            # iphone-14-pro-max-hevc-4k-dolby-vision-59.99fps-vertical.MOV
            # iphone-14-pro-max-hevc-4k-sdr-29.99fps-vertical.MOV
            # iphone-14-pro-max-hevc-1080p-dolby-vision-59.89fps-landscape.MOV
            # iphone-14-pro-max-pro-res-1080p-hdr-29.98fps-landscape.MOV

            if audio_stream:
                metadata.update(
                    {
                        "sample_rate": int(audio_stream.get("sample_rate", None)),
                        "channels": str(audio_stream.get("channels", None)),
                        "codec": audio_stream.get("codec_long_name", None),
                        "bit_depth": audio_stream.get("bits_per_sample", None),
                    }
                )

            if video_stream:
                width = video_stream.get("width")
                height = video_stream.get("height")
                metadata.update(
                    {
                        "video_codec": video_stream.get("codec_long_name", None),
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
