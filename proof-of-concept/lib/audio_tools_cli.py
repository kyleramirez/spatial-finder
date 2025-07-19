from pathlib import Path
from tqdm import tqdm
from typing import List
from lib.utils import SUPPORTED_AUDIO_FORMATS, SUPPORTED_VIDEO_FORMATS, SUPPORTED_CONTAINER_FORMATS
from lib.audio_database import AudioDatabase
from lib.audio_processor import AudioProcessor


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
