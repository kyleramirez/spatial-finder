#!/bin/bash

# Suppress ffmpeg banner and get formats
formats=$(ffmpeg -formats -hide_banner 2>/dev/null | grep -E "^ [DE ]{2}" | awk '{print $2}')

# Print CSV header
echo "extension,audio_supported,video_supported"

# Loop through each format
while IFS= read -r fmt; do
    # Get muxer info for the format
    info=$(ffmpeg -hide_banner -h muxer="$fmt" 2>/dev/null)
    
    # Check for audio and video support
    audio_support=$(echo "$info" | grep -q "Default audio codec" && echo "true" || echo "false")
    video_support=$(echo "$info" | grep -q "Default video codec" && echo "true" || echo "false")
    
    # Extract common extensions, default to format name if not found
    ext=$(echo "$info" | grep "Common extensions:" | sed -n 's/.*Common extensions: \([^.]*\)\..*/\1/p' | head -n 1)
    ext=${ext:-$fmt}  # Use format name as fallback if no extension is found
    
    # Print CSV row with the first common extension (or format name)
    echo "$ext,$audio_support,$video_support"
done <<< "$formats"