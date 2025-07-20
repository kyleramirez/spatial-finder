import duckdb
import hashlib
from .utils import DB_PATH


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
                basename TEXT NOT NULL,
                bit_depth TEXT,
                bitrate INTEGER,
                channels TEXT,
                codec TEXT,
                codec_long TEXT,
                color_space TEXT,
                container_format TEXT,
                container_format_long TEXT,
                creation_date TIMESTAMP,
                duration REAL,
                encoder TEXT,
                extension TEXT NOT NULL,
                frame_rate REAL,
                generalized_location TEXT,
                has_video BOOLEAN DEFAULT false,
                ingest_status TEXT DEFAULT 'QUEUED',
                latitude REAL,
                longitude REAL,
                mime TEXT,
                path TEXT NOT NULL,
                profile TEXT,
                resolution TEXT,
                sample_rate INTEGER,
                strategy TEXT DEFAULT 'local_disk',
                video_bit_depth TEXT,
                video_bitrate INTEGER,
                video_codec TEXT,
                video_codec_long TEXT,
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
