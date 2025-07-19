# The Plan

## Phases
1. Build bare-bones proof-of-concept, Python and usage via command line and /test-audio directory
2. Build out front-end of Tauri app
3. Build Rust version that mirrors command line functionality
4. Wire up front-end to Rust via Tauri


## Planned features for command line proof of concept
1. Ingest audio by given individual file or by recursing through chosen directory

       ./audible_tools.py add ./audio-samples/ds40/DS400292.WMA
       ./audible_tools.py add -R ./audio-samples

   This will ingest the audio found at those locations into the database. It must return a proper exit code and log stats on what it did. Ultimately, this will queue a list of files to be processed. So if the file is an audio file, save it in `audio_files`, and queue the job to process it. This is also how to rebuild resources in the system if the information is now stale. All newly found files will be created, any no-longer-existing files will be cleaned from the database, and all existing files will be skipped unless they have changed, e.g. if their deterministically created uuid is different.

1. Check out the status of the running jobs, and other relevant totals

       ./audible_tools.py status


1. Browse files by stats, speakers, timeline, text, processing status

       ./audible_tools.py ls ./audio-samples

   This could come back with a table of information such as:
   |File|Added|Status|Voices|Transcript|Info|
   |-|-|-|-|-|-|
   |interview1.wav|Yes|Processed|Tim, Shelly|Truncated too la..|Duration: 24:34|
   |interview2.wav|Yes|Processing|-|-|-|
   |interview3.wav|No|None|-|-|-|

1. List, name and merge speakers, listen to most representative (centroid) audio clip

       ./audible_tools.py voices list
       # Lists known voices and their most relevant clip location I can play (ideally in ogg)

       ./audible_tools.py voices show person1
       # Shows speaker and associated clips

       ./audible_tools.py voices rename person1 Mike
       # Renames voices

       ./audible_tools.py voices merge Mike person25 --dry-run
       # Changes all records associated with person25 to be associated with Mike, deletes person25. Use transaction if capable, can do a dry run

1. Search by voice, plain english

       ./audible_tools.py voice search Mike
       ./audible_tools.py search "Dinner plans"

   This will show listenable clips of Mike or clips most related to dinner plans with a configurable threshold.

1. Export transcript(s) as sidecar to original audio file by individual file or for entire directory recursively, overwriting as needed

       ./audible_tools.py export ./audio-samples/ds40/DS400292.WMA
       ./audible_tools.py export -R ./audio-samples

1. Erase all managed data

       ./audible_tools.py reset

   This will prompt for a confirmation to destroy all internal databases and restart from scratch
1. You can also remove the database records for the files manually without rebuilding or deleting the files themselves similar to how you added them:

       ./audible_tools.py rm ./audio-samples/ds40/DS400292.WMA
       ./audible_tools.py rm -R ./audio-samples

## Implementation notes
- Include any instructions it takes to initialize such as things that aren't covered by pip install but ideally keep the OS dependencies very low and ask first. Create a Dockerfile if needed to document / contain external dependencies.
- Use requirements.txt to add dependencies
- When building in Python, work inside the ./proof-of-concept directory.
  - Initialize the environment if it hasn't yet been

        python -m venv virtualenv
  - Activate the virtual environment

        source virtualenv/bin/activate
  - Install and maintain dependencies

        pip install -r requirements.txt
  - The audible_tools.py must call the main.py

        ./audible_tools.py <command> <args>
- Write sensible python tests and execute them as you go to ensure you're staying on track
- Plan on a completely non-destructive interface for performing operations except for when overwriting previously generated transcripts as sidecar files. If they export original-audio.mp3, the exported file should be something like original-audio-exported-{timestamp-of-export}.mp3 exported to the same location by default but also able to specify a destination, and under no circumstance can it overwrite original files
- Plan on encountering 32-bit float audio, so will need to convert to a usable format
- Plan on encountering .wma files, or other various files. Ideally it can handle any audio format its libs can handle, but can still comprehensively list them
- Plan on possibility of built-in audio pre-filtering to normalize volume, bring out voices when necessary, etc.
- When the user is going to listen to something they found, the application should be ready to respond with a file URL they can open in VLC, and if there's captions associated with it (pretty much always should be), those can be read from the file or sidecar file using VLC. Even better if they don't have to copy/paste anything and the command line can open it all up on request
- When a search result is selected, the result should not start out mid-sentence or mid-sound, but be a clip of the nearest whole sentence or start of sound, especially important when doing the rolling window CLIP embeddings
- When listening to audio clips via the command line or future web interface, plan on everything being converted to OGG / Vorbis for max compatibility regardless of the original format, and the browser can maintain a cache of those files.
- Use consistent naming / casing / units as much as possible, correcting my plan as needed
- One tool I found can populate audio / video-related fields with something like `ffprobe -v quiet -print_format json -show_format -show_streams input.mp4`
- I previously was able to convert wma to wav using a script in scripts/convert-wma-wav.sh

## Tool / model selection
- Whisper for generating audio transcripts
- pyannote for speaker diarization
- duckdb for persisent storage
- FAISS for similarity search
- Sentence transformers
- Any other mentioned models

## Proposed schema
- `audible_files`: metadata (audio only at first, video later)
  - `uuid`: Deterministic for file
  - `strategy`: (only `local_disk` at first, Dropbox or more later)
  - `path`: File location
  - `basename`: File basename
  - `extension`: e.g. "mp3"
  - `mime`: MIME type of file, e.g. "audio/mpeg", "video/mp4"
  - `sample_rate`: in khz, e.g. 48000
  - `bitrate`: in kbps, e.g. 256000
  - `bit_depth`: e.g. "16", "24", "32", or "32f"
  - `channels`: Stereo vs mono vs 5.1, e.g. "2", "1", "5.1"
  - `codec`: Clarifies stream content beyond MIME, e.g. "pcm_s32le", "aac", "flac"
  - `has_video`: Default false. True when the audio is part of a video file in a future feature
  - `video_codec`: e.g. "h264", "vp9", "prores"
  - `video_resolution`: e.g. "1920x1080"
  - `frame_rate`: REAL e.g. 29.97
  - `container_format`: mp4, mkv, mov
  - `creation_date`: Timestamp for when file was recorded
  - `generalized_location`: Optional text field for recording general location for future XMP / ID3 / Vorbis Comments export, example value: "Chicago" or "Dale's house"
  - `latitude`: Optional value for recording location for future XMP / ID3 / Vorbis Comments export
  - `longitude`: Optional value for recording location for future XMP / ID3 / Vorbis Comments export
  - `duration`: in seconds for audio / video only
  - `ingest_status`: QUEUED, WORKING, COMPLETE, FAILED
- `voices`: Speaker diarization output, represents a single person, can be merged with other found voices
  - `id`
  - `display_name`: Default to "Unknown"
  - `best_verbalization_id` Of all the verbalizations connected to this voice, this one is the most representative, i.e. the centroid of the voice's similarity cluster
- `verbalizations`: Table for storing Whisper-found transcript segments
  - `voice_id`: required
  - `id`
  - `audible_file_uuid`
  - `start_byte`: start byte of the start time
  - `end_byte`: end byte for the end of the event
  - `start_time`: in seconds from beginning of file
  - `duration`: in seconds
  - `label`: Raw text of transcript utterance
  - `label_confidence`: Whisper model confidence normalized 0.0 - 1.0
  - `label_embedding`: Sentence transformer embeddings of `label` for similarity search
  - `label_model`: e.g. "sentence-t5-large"
  - `voice_confidence`: Pyannote confidence normalized 0.0 - 1.0
  - `voice_embedding`: Pyannote embedding for similarity search
  - `voice_model`: e.g. "wespeaker_en_voxceleb_CAM"
- `nonverbal_labels`: For model or manually labeled non-speech sounds
  - `id`
  - `audible_file_uuid`
  - `start_byte`: start byte of the start time
  - `end_byte`: end byte for the end of the event
  - `start_time`: in seconds from beginning of file
  - `duration`: in seconds
  - `label`: Labeled sound (e.g. "bird chirping")
  - `source`: MODEL, MANUAL
  - `kind`:
    - `NOTE`: Subjective or semantic interpretation, e.g. "Conversation becomes tense here", or "Change of topic from weather to politics"
    - `DESCRIPTION`: High-level description of the segment, e.g. "Discussion about where to eat next", "bird chirping"
    - `TRANSLATION`: Translated from another language
  - `confidence`: Model confidence normalized 0.0 - 1.0, required if source is not MANUAL
  - `embedding`: CLAP or PANNs vector, required if source is not MANUAL
  - `embedding_model`: CLAP or PANNs, required if source is not MANUAL
  - `include_in_export`: Boolean for sounds that should stay here, but not be present in export, default `false`
- `silents`: Model labeled periods of silence
  - `id`
  - `audible_file_uuid`
  - `start_byte`: start byte of the start time
  - `end_byte`: end byte for the end of the event
  - `start_time`: in seconds from beginning of file
  - `duration`: in seconds
  - `confidence`: Model confidence normalized 0.0 - 1.0
  - `detector`: e.g. "silero-vad", "librosa", etc.
  - `detector_version`: 
- `audible_embeddings`: Embeddings for 5-second sound windows overlapping by 2.5s, excluding periods of silence
  - `id`
  - `audible_file_uuid`
  - `start_byte`: start byte of the start time
  - `end_byte`: end byte for the end of the speech
  - `start_time`: in seconds from beginning of file
  - `duration` in seconds
  - `embedding`: CLIP embedding of audible period
  - `embedding_model`: e.g. "clip-vit-b-32"
  - `segment_index`: 0-based index of overlapping windows for reconstructing embedding coverage
- `exports`
  - `id`
  - `status`: QUEUED, WORKING, COMPLETE, FAILED
  - `strategy`: SRT, VTT
  - `created_at`: timestamp
  - `updated_at`: timestamp
- `export_voices`: Any custom label to use in the SRT/VTT for a voice different from the `voices.display_name`
  - `voice_id`: ID of the original voice present in the export
  - `display_name`: Override name to use for the export
- `audible_faiss_indexes`: Helpful for knowing when the in-memory index is stale or needs rebuild
  - `id`
  - `type`: "audio", "utterance", etc.
  - `strategy`: "HNSW", "IVF", etc.
  - `index_path`: e.g. "/tmp/audio_index_2024.idx"
  - `created_at` TIMESTAMP
  - `updated_at` TIMESTAMP
