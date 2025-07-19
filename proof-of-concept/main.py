#!/usr/bin/env python3
"""
Audible Tools - Audio Processing and Transcription Tool
Command line interface for audio file processing, transcription, and search.
"""
import click
from dotenv import load_dotenv
from lib.audio_tools_cli import AudioToolsCLI

load_dotenv()


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
