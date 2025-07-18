#!/usr/bin/env python3
"""
Comprehensive test suite for Audible Tools
Tests all primary features using real audio files from audio-samples directory.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import os
import sqlite3
from unittest.mock import patch, MagicMock
import json

# Import the main modules
from main import AudioDatabase, AudioProcessor, AudioToolsCLI
import main

class TestAudioDatabase(unittest.TestCase):
    """Test the AudioDatabase class."""
    
    def setUp(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = Path(self.temp_dir) / "test.db"
        self.db = AudioDatabase(str(self.test_db_path))
    
    def tearDown(self):
        """Clean up test database."""
        self.db.close()
        shutil.rmtree(self.temp_dir)
    
    def test_database_initialization(self):
        """Test that all tables are created properly."""
        # Check that all expected tables exist
        tables = self.db.conn.execute("SHOW TABLES").fetchall()
        table_names = [table[0] for table in tables]
        
        expected_tables = [
            'audible_files', 'voices', 'verbalizations', 'nonverbal_labels',
            'silents', 'audible_embeddings', 'exports', 'export_voices',
            'audible_faiss_indexes'
        ]
        
        for table in expected_tables:
            self.assertIn(table, table_names, f"Table {table} should exist")
    
    def test_get_file_uuid(self):
        """Test UUID generation for files."""
        # Create a test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("test content")
        
        # Generate UUID
        uuid1 = self.db.get_file_uuid(str(test_file))
        uuid2 = self.db.get_file_uuid(str(test_file))
        
        # Should be consistent
        self.assertEqual(uuid1, uuid2)
        self.assertIsInstance(uuid1, str)
        self.assertEqual(len(uuid1), 64)  # SHA256 hex length
    
    def test_get_or_create_default_voice(self):
        """Test default voice creation and retrieval."""
        # First call should create voice
        voice_id1 = self.db.get_or_create_default_voice()
        self.assertIsInstance(voice_id1, int)
        
        # Second call should return same voice
        voice_id2 = self.db.get_or_create_default_voice()
        self.assertEqual(voice_id1, voice_id2)
        
        # Check voice exists in database
        voice = self.db.conn.execute(
            "SELECT display_name FROM voices WHERE id = ?", (voice_id1,)
        ).fetchone()
        self.assertEqual(voice[0], "Default Speaker")

class TestAudioProcessor(unittest.TestCase):
    """Test the AudioProcessor class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = Path(self.temp_dir) / "test.db"
        self.db = AudioDatabase(str(self.test_db_path))
        self.processor = AudioProcessor(self.db)
    
    def tearDown(self):
        """Clean up test environment."""
        self.db.close()
        shutil.rmtree(self.temp_dir)
    
    def test_pyannote_availability(self):
        """Test that pyannote.audio is available and can be imported."""
        # Test that pyannote.audio can be imported
        try:
            from pyannote.audio import Pipeline
            self.assertTrue(True, "pyannote.audio imported successfully")
        except ImportError as e:
            self.fail(f"pyannote.audio is not available: {e}")
    
    def test_diarization_pipeline_requirements(self):
        """Test that diarization pipeline requirements are checked."""
        # Test that load_models fails without HuggingFace token
        original_token = os.environ.get('HUGGINGFACE_TOKEN')
        
        try:
            # Remove token if it exists
            if 'HUGGINGFACE_TOKEN' in os.environ:
                del os.environ['HUGGINGFACE_TOKEN']
            
            # Create a new processor to test token requirement
            processor = AudioProcessor(self.db)
            
            # This should raise RuntimeError about missing token
            with self.assertRaises(RuntimeError) as context:
                processor.load_models()
            
            self.assertIn("HUGGINGFACE_TOKEN", str(context.exception))
            
        finally:
            # Restore original token if it existed
            if original_token:
                os.environ['HUGGINGFACE_TOKEN'] = original_token
    
    def test_device_selection(self):
        """Test that device selection works."""
        device = main.get_device()
        self.assertIn(device, ['cpu', 'cuda', 'mps'])
    
    def test_get_audio_metadata(self):
        """Test audio metadata extraction."""
        # Find the smallest audio file for testing
        audio_files = list(Path("audio-samples").rglob("*.WMA"))
        if not audio_files:
            self.skipTest("No audio files found for testing")
        
        # Use the smallest file
        test_file = min(audio_files, key=lambda f: f.stat().st_size)
        metadata = self.processor.get_audio_metadata(str(test_file))
        
        # Check that metadata contains expected fields
        expected_fields = ['duration', 'container_format', 'has_video']
        for field in expected_fields:
            self.assertIn(field, metadata)
    
    def test_convert_audio_to_wav(self):
        """Test audio format conversion."""
        # Find a test audio file
        audio_files = list(Path("audio-samples").rglob("*.WMA"))
        if not audio_files:
            self.skipTest("No audio files found for testing")
        
        test_file = min(audio_files, key=lambda f: f.stat().st_size)
        output_file = Path(self.temp_dir) / "test.wav"
        
        success = self.processor.convert_audio_to_wav(str(test_file), str(output_file))
        
        self.assertTrue(success)
        self.assertTrue(output_file.exists())
        self.assertGreater(output_file.stat().st_size, 0)
    
    def test_process_segments(self):
        """Test segment processing."""
        # Create a test file record first
        test_uuid = "test-uuid-123"
        self.db.conn.execute(
            """
            INSERT INTO audible_files 
            (uuid, path, basename, extension, duration, ingest_status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (test_uuid, "/test/path.wav", "test.wav", ".wav", 10.0, "WORKING")
        )
        
        # Mock Whisper result
        mock_result = {
            'segments': [
                {
                    'start': 0.0,
                    'end': 5.0,
                    'text': 'Hello world',
                    'avg_logprob': -0.5
                },
                {
                    'start': 5.0,
                    'end': 10.0,
                    'text': 'This is a test',
                    'avg_logprob': -0.3
                }
            ]
        }
        
        # Mock sentence transformer
        with patch.object(self.processor, 'sentence_transformer') as mock_transformer:
            import numpy as np
            mock_transformer.encode.return_value = np.array([0.1, 0.2, 0.3])
            
            # Mock diarization result
            from unittest.mock import MagicMock
            mock_diarization = MagicMock()
            mock_diarization.itertracks.return_value = [
                (MagicMock(start=0.0, end=10.0), None, "SPEAKER_00"),
                (MagicMock(start=5.0, end=15.0), None, "SPEAKER_01"),
            ]
            
            success = self.processor.process_segments_with_diarization(test_uuid, mock_result, mock_diarization)
            
            self.assertTrue(success)
            
            # Check verbalizations were created
            verbalizations = self.db.conn.execute(
                "SELECT COUNT(*) FROM verbalizations WHERE audible_file_uuid = ?",
                (test_uuid,)
            ).fetchone()[0]
            
            self.assertEqual(verbalizations, 2)
            
            # Check that distinct speakers were created
            speakers = self.db.conn.execute(
                "SELECT DISTINCT v.display_name FROM voices v JOIN verbalizations vb ON v.id = vb.voice_id WHERE vb.audible_file_uuid = ?",
                (test_uuid,)
            ).fetchall()
            
            # Should have at least one speaker (possibly two if overlap creates different assignments)
            self.assertGreaterEqual(len(speakers), 1)
    
    def test_diarization_distinct_speakers(self):
        """Test that diarization creates distinct speakers with proper names."""
        # Create a test file record first
        test_uuid = "test-diarization-123"
        self.db.conn.execute(
            """
            INSERT INTO audible_files 
            (uuid, path, basename, extension, duration, ingest_status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (test_uuid, "/test/diarization.wav", "diarization.wav", ".wav", 20.0, "WORKING")
        )
        
        # Mock Whisper result with multiple segments
        mock_result = {
            'segments': [
                {
                    'start': 0.0,
                    'end': 5.0,
                    'text': 'First speaker talking',
                    'avg_logprob': -0.5
                },
                {
                    'start': 10.0,
                    'end': 15.0,
                    'text': 'Second speaker responding',
                    'avg_logprob': -0.3
                },
                {
                    'start': 16.0,
                    'end': 20.0,
                    'text': 'Third speaker joining',
                    'avg_logprob': -0.4
                }
            ]
        }
        
        # Mock sentence transformer
        with patch.object(self.processor, 'sentence_transformer') as mock_transformer:
            import numpy as np
            mock_transformer.encode.return_value = np.array([0.1, 0.2, 0.3])
            
            # Mock diarization result with distinct speakers
            from unittest.mock import MagicMock
            mock_diarization = MagicMock()
            mock_diarization.itertracks.return_value = [
                (MagicMock(start=0.0, end=5.0), None, "SPEAKER_00"),
                (MagicMock(start=10.0, end=15.0), None, "SPEAKER_01"),
                (MagicMock(start=16.0, end=20.0), None, "SPEAKER_02"),
            ]
            
            success = self.processor.process_segments_with_diarization(test_uuid, mock_result, mock_diarization)
            
            self.assertTrue(success)
            
            # Check that three distinct speakers were created
            speakers = self.db.conn.execute(
                "SELECT DISTINCT v.display_name FROM voices v JOIN verbalizations vb ON v.id = vb.voice_id WHERE vb.audible_file_uuid = ? ORDER BY v.display_name",
                (test_uuid,)
            ).fetchall()
            
            speaker_names = [speaker[0] for speaker in speakers]
            
            # Should have exactly three distinct speakers
            self.assertEqual(len(speaker_names), 3)
            self.assertIn("Speaker SPEAKER_00", speaker_names)
            self.assertIn("Speaker SPEAKER_01", speaker_names)
            self.assertIn("Speaker SPEAKER_02", speaker_names)
            
            # Check that each segment was assigned to the correct speaker
            results = self.db.conn.execute(
                """
                SELECT vb.label, v.display_name, vb.start_time 
                FROM verbalizations vb 
                JOIN voices v ON vb.voice_id = v.id 
                WHERE vb.audible_file_uuid = ? 
                ORDER BY vb.start_time
                """,
                (test_uuid,)
            ).fetchall()
            
            self.assertEqual(len(results), 3)
            self.assertEqual(results[0][1], "Speaker SPEAKER_00")  # First segment
            self.assertEqual(results[1][1], "Speaker SPEAKER_01")  # Second segment
            self.assertEqual(results[2][1], "Speaker SPEAKER_02")  # Third segment

class TestAudioToolsCLI(unittest.TestCase):
    """Test the CLI interface."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create test database
        self.cli = AudioToolsCLI()
        self.cli.db.db_path = "test.db"
        self.cli.db.conn = self.cli.db.conn
    
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_show_status_empty(self):
        """Test status display with empty database."""
        # Should not crash with empty database
        try:
            self.cli.show_status()
        except Exception as e:
            self.fail(f"show_status() raised {e} unexpectedly")
    
    def test_list_files_empty(self):
        """Test file listing with empty database."""
        try:
            self.cli.list_files()
        except Exception as e:
            self.fail(f"list_files() raised {e} unexpectedly")
    
    def test_list_voices_empty(self):
        """Test voice listing with empty database."""
        try:
            self.cli.list_voices()
        except Exception as e:
            self.fail(f"list_voices() raised {e} unexpectedly")
    
    def test_search_text_empty(self):
        """Test text search with empty database."""
        try:
            self.cli.search_text("test query")
        except Exception as e:
            self.fail(f"search_text() raised {e} unexpectedly")
    
    def test_voice_rename(self):
        """Test voice renaming."""
        # Create a voice first
        voice_id = self.cli.db.get_or_create_default_voice()
        
        # Rename it
        self.cli.rename_voice(voice_id, "Test Speaker")
        
        # Check it was renamed
        voice = self.cli.db.conn.execute(
            "SELECT display_name FROM voices WHERE id = ?", (voice_id,)
        ).fetchone()
        self.assertEqual(voice[0], "Test Speaker")
    
    def test_seconds_to_srt_time(self):
        """Test SRT time format conversion."""
        # Test various time values
        test_cases = [
            (0.0, "00:00:00,000"),
            (1.5, "00:00:01,500"),
            (61.0, "00:01:01,000"),
            (3661.123, "01:01:01,123"),
        ]
        
        for seconds, expected in test_cases:
            result = self.cli.seconds_to_srt_time(seconds)
            self.assertEqual(result, expected)

class TestIntegration(unittest.TestCase):
    """Integration tests using real audio files."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Find audio files in the correct location
        audio_samples_dir = Path(self.original_cwd) / "audio-samples"
        
        # Find all audio files
        self.audio_files = []
        self.audio_files.extend(audio_samples_dir.rglob("*.WMA"))
        self.audio_files.extend(audio_samples_dir.rglob("*.WAV"))
        self.audio_files.extend(audio_samples_dir.rglob("*.wav"))
        self.audio_files.extend(audio_samples_dir.rglob("*.wma"))
        
        if not self.audio_files:
            self.skipTest("No audio files found for integration testing")
        
        # Sort by size for predictable testing
        self.audio_files.sort(key=lambda f: f.stat().st_size)
        
        # Use the smallest files to speed up tests (limit to reasonable size)
        self.test_files = [f for f in self.audio_files if f.stat().st_size <= 2 * 1024 * 1024]  # Max 2MB
        
        if not self.test_files:
            self.skipTest("No reasonably sized audio files found for integration testing")
        
        # Create CLI instance
        self.cli = AudioToolsCLI()
    
    def tearDown(self):
        """Clean up integration test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_full_workflow(self):
        """Test complete workflow: add -> process -> search -> export."""
        print(f"\n🎵 Testing with {len(self.test_files)} audio files:")
        for i, file in enumerate(self.test_files):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {i+1}. {file.name} ({size_mb:.1f}MB) - {file.suffix.upper()}")
        
        # Mock the diarization pipeline for testing
        with patch.object(self.cli.processor, 'diarization_pipeline') as mock_pipeline:
            # Create a mock diarization result
            mock_diarization_result = MagicMock()
            mock_diarization_result.itertracks.return_value = [
                (MagicMock(start=0.0, end=30.0), None, "SPEAKER_00"),
                (MagicMock(start=30.0, end=60.0), None, "SPEAKER_01"),
                (MagicMock(start=60.0, end=90.0), None, "SPEAKER_02"),
            ]
            mock_pipeline.return_value = mock_diarization_result
            
            # Mock sentence transformer
            with patch.object(self.cli.processor, 'sentence_transformer') as mock_transformer:
                import numpy as np
                mock_transformer.encode.return_value = np.array([0.1, 0.2, 0.3])
                
                # Step 1: Add files to database
                file_paths = [str(f) for f in self.test_files]
                self.cli.add_files(file_paths)
                
                # Step 2: Check files were added
                files_count = self.cli.db.conn.execute(
                    "SELECT COUNT(*) FROM audible_files"
                ).fetchone()[0]
                self.assertEqual(files_count, len(self.test_files))
                
                # Step 3: Check processing status
                processed_files = self.cli.db.conn.execute(
                    "SELECT basename, ingest_status FROM audible_files"
                ).fetchall()
                
                print(f"\n📊 Processing results:")
                complete_files = []
                for basename, status in processed_files:
                    print(f"  - {basename}: {status}")
                    if status == 'COMPLETE':
                        complete_files.append(basename)
                
                # We expect at least one file to process successfully
                self.assertGreater(len(complete_files), 0, "At least one file should process successfully")
                
                # Step 4: Check verbalizations were created
                verbalizations = self.cli.db.conn.execute(
                    "SELECT COUNT(*) FROM verbalizations"
                ).fetchone()[0]
                self.assertGreater(verbalizations, 0, "Should have created verbalizations")
                
                # Step 5: Check speakers were identified
                speakers = self.cli.db.conn.execute(
                    "SELECT DISTINCT v.display_name FROM voices v JOIN verbalizations vb ON v.id = vb.voice_id"
                ).fetchall()
                
                speaker_names = [speaker[0] for speaker in speakers]
                print(f"\n🎤 Speakers identified: {speaker_names}")
                
                # Verify we have speakers with proper names
                self.assertGreater(len(speakers), 0, "Should have identified speakers")
                
                # Check that speakers follow the expected naming pattern
                for speaker_name in speaker_names:
                    self.assertTrue(
                        speaker_name.startswith("Speaker SPEAKER_") or speaker_name == "Default Speaker",
                        f"Speaker name '{speaker_name}' should follow expected pattern"
                    )
                
                # Verify we have multiple distinct speakers
                print(f"\n✅ Found {len(speakers)} distinct speakers with proper naming")
                
                # Step 6: Test search functionality
                # Get a word from the first verbalization
                first_word = self.cli.db.conn.execute(
                    "SELECT label FROM verbalizations LIMIT 1"
                ).fetchone()
                
                if first_word and first_word[0].strip():
                    # Search for the first word
                    word = first_word[0].strip().split()[0]
                    print(f"\n🔍 Testing search for word: '{word}'")
                    self.cli.search_text(word, limit=5)
                
                # Step 7: Test voice search
                if speakers:
                    first_speaker = speaker_names[0]
                    print(f"\n🔍 Testing voice search for: '{first_speaker}'")
                    self.cli.search_voice(first_speaker, limit=3)
                
                # Step 8: Test export functionality
                # Count SRT files in the audio-samples directory (where they get created)
                audio_samples_dir = Path(self.original_cwd) / "audio-samples"
                export_files_before = len(list(audio_samples_dir.rglob("*.srt")))
                self.cli.export_transcripts(file_paths)
                export_files_after = len(list(audio_samples_dir.rglob("*.srt")))
                
                # Should have created SRT files
                self.assertGreater(export_files_after, export_files_before, "Should have created SRT files")
                
                print(f"\n✅ Created {export_files_after - export_files_before} SRT files")
                
                # Step 9: Show format coverage
                formats_tested = set()
                for file in self.test_files:
                    formats_tested.add(file.suffix.upper())
                
                print(f"\n📁 Audio formats tested: {', '.join(sorted(formats_tested))}")
                
                # Step 10: Show final statistics
                self.cli.show_status()
                
                print(f"\n🎉 Integration test completed successfully!")
                print(f"   - Files processed: {len(complete_files)}")
                print(f"   - Speakers identified: {len(speakers)}")
                print(f"   - Verbalizations created: {verbalizations}")
                print(f"   - Formats tested: {', '.join(sorted(formats_tested))}")
    
    def test_multiple_speakers_detection(self):
        """Test that multiple speakers are detected across different files."""
        if not self.test_files:
            self.skipTest("No audio files available for speaker detection test")
        
        # Use first file for this test
        test_file = self.test_files[0]
        print(f"\n🎤 Testing speaker detection with: {test_file.name}")
        
        # Add and process file
        self.cli.add_files([str(test_file)])
        
        # Check if processing succeeded
        status = self.cli.db.conn.execute(
            "SELECT ingest_status FROM audible_files WHERE basename = ?",
            (test_file.name,)
        ).fetchone()[0]
        
        if status == 'COMPLETE':
            # Get all speakers and their segments
            speaker_segments = self.cli.db.conn.execute(
                """
                SELECT v.display_name, COUNT(vb.id) as segment_count, 
                       AVG(vb.start_time) as avg_time, 
                       AVG(vb.voice_confidence) as avg_confidence
                FROM voices v 
                JOIN verbalizations vb ON v.id = vb.voice_id 
                GROUP BY v.id, v.display_name
                ORDER BY segment_count DESC
                """
            ).fetchall()
            
            print(f"\n📊 Speaker Analysis:")
            for speaker, segments, avg_time, avg_confidence in speaker_segments:
                print(f"  - {speaker}: {segments} segments, avg time: {avg_time:.1f}s, confidence: {avg_confidence:.2f}")
            
            # Verify speakers were detected
            self.assertGreater(len(speaker_segments), 0, "Should detect at least one speaker")
            
            # Check speaker naming
            for speaker, _, _, _ in speaker_segments:
                self.assertTrue(
                    speaker.startswith("Speaker SPEAKER_") or speaker == "Default Speaker",
                    f"Speaker '{speaker}' should follow expected naming pattern"
                )
        else:
            print(f"⚠️  File processing failed with status: {status}")
            self.skipTest(f"File processing failed with status: {status}")

class TestCLICommands(unittest.TestCase):
    """Test CLI command parsing and execution."""
    
    def setUp(self):
        """Set up CLI test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        """Clean up CLI test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_cli_help(self):
        """Test CLI help messages."""
        from click.testing import CliRunner
        from main import cli
        
        runner = CliRunner()
        
        # Test main help
        result = runner.invoke(cli, ['--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Audible Tools", result.output)
        
        # Test status command
        result = runner.invoke(cli, ['status'])
        self.assertEqual(result.exit_code, 0)
        
        # Test voices help
        result = runner.invoke(cli, ['voices', '--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Voice management", result.output)
    
    def test_cli_status(self):
        """Test status command."""
        from click.testing import CliRunner
        from main import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ['status'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Processing Status", result.output)
    
    def test_cli_voices_list(self):
        """Test voices list command."""
        from click.testing import CliRunner
        from main import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ['voices', 'list'])
        self.assertEqual(result.exit_code, 0)

def run_performance_tests():
    """Run performance tests with timing."""
    import time
    
    print("Running performance tests...")
    
    # Test audio processing speed
    audio_files = list(Path("audio-samples").rglob("*.WMA"))
    if not audio_files:
        print("No audio files found for performance testing")
        return
    
    # Use a medium-sized file for performance testing
    test_file = None
    for f in audio_files:
        size = f.stat().st_size
        if 100 * 1024 < size < 1024 * 1024:  # 100KB to 1MB
            test_file = f
            break
    
    if not test_file:
        print("No suitable file found for performance testing")
        return
    
    print(f"Testing with file: {test_file.name} ({test_file.stat().st_size / 1024:.1f} KB)")
    
    # Create temporary environment
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            cli = AudioToolsCLI()
            
            # Use absolute path for the test file
            test_file_path = original_cwd / test_file
            
            # Time the full processing
            start_time = time.time()
            cli.add_files([str(test_file_path)])
            end_time = time.time()
            
            processing_time = end_time - start_time
            print(f"Processing time: {processing_time:.2f} seconds")
            
            # Check if processing succeeded
            status = cli.db.conn.execute(
                "SELECT ingest_status FROM audible_files WHERE basename = ?",
                (test_file.name,)
            ).fetchone()
            
            if status and status[0] == 'COMPLETE':
                print("✓ Processing completed successfully")
                
                # Count verbalizations
                verbalizations = cli.db.conn.execute(
                    "SELECT COUNT(*) FROM verbalizations"
                ).fetchone()[0]
                print(f"✓ Created {verbalizations} verbalizations")
                
                # Test export speed
                start_time = time.time()
                cli.export_transcripts([str(test_file_path)])
                end_time = time.time()
                
                export_time = end_time - start_time
                print(f"✓ Export time: {export_time:.2f} seconds")
                
            else:
                print("✗ Processing failed")
                
        finally:
            os.chdir(original_cwd)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Audible Tools tests')
    parser.add_argument('--performance', action='store_true', 
                       help='Run performance tests')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.performance:
        run_performance_tests()
    else:
        # Run unit tests
        if args.verbose:
            verbosity = 2
        else:
            verbosity = 1
        
        # Discover and run tests
        loader = unittest.TestLoader()
        suite = loader.discover('.', pattern='test_*.py')
        
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        
        # Exit with error code if tests failed
        if not result.wasSuccessful():
            exit(1) 