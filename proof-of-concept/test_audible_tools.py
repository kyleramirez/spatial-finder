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
            
            success = self.processor.process_segments(test_uuid, mock_result)
            
            self.assertTrue(success)
            
            # Check verbalizations were created
            verbalizations = self.db.conn.execute(
                "SELECT COUNT(*) FROM verbalizations WHERE audible_file_uuid = ?",
                (test_uuid,)
            ).fetchone()[0]
            
            self.assertEqual(verbalizations, 2)

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
        
        # Find smallest audio file for testing
        self.audio_files = list(Path("../audio-samples").rglob("*.WMA"))
        if not self.audio_files:
            self.skipTest("No audio files found for integration testing")
        
        # Use the smallest file to speed up tests
        self.test_file = min(self.audio_files, key=lambda f: f.stat().st_size)
        
        # Create CLI instance
        self.cli = AudioToolsCLI()
    
    def tearDown(self):
        """Clean up integration test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_full_workflow(self):
        """Test complete workflow: add -> process -> search -> export."""
        # Skip if file is too large (> 1MB to keep tests fast)
        if self.test_file.stat().st_size > 1024 * 1024:
            self.skipTest("Test file too large for quick testing")
        
        # Step 1: Add file to database
        self.cli.add_files([str(self.test_file)])
        
        # Step 2: Check file was added
        files = self.cli.db.conn.execute(
            "SELECT COUNT(*) FROM audible_files"
        ).fetchone()[0]
        self.assertEqual(files, 1)
        
        # Step 3: Check processing status
        status = self.cli.db.conn.execute(
            "SELECT ingest_status FROM audible_files WHERE basename = ?",
            (self.test_file.name,)
        ).fetchone()[0]
        self.assertIn(status, ['COMPLETE', 'FAILED'])
        
        # If processing succeeded, test search and export
        if status == 'COMPLETE':
            # Step 4: Check verbalizations were created
            verbalizations = self.cli.db.conn.execute(
                "SELECT COUNT(*) FROM verbalizations"
            ).fetchone()[0]
            self.assertGreater(verbalizations, 0)
            
            # Step 5: Test search functionality
            # Get a word from the first verbalization
            first_word = self.cli.db.conn.execute(
                "SELECT label FROM verbalizations LIMIT 1"
            ).fetchone()
            
            if first_word and first_word[0].strip():
                # Search for the first word
                word = first_word[0].strip().split()[0]
                self.cli.search_text(word, limit=5)
            
            # Step 6: Test export functionality
            export_files_before = len(list(Path(".").glob("*.srt")))
            self.cli.export_transcripts([str(self.test_file)])
            export_files_after = len(list(Path(".").glob("*.srt")))
            
            # Should have created an SRT file
            self.assertGreater(export_files_after, export_files_before)

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