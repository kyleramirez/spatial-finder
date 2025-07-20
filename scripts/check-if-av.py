import mimetypes, subprocess, re

# Put a set here of the formats you want to classify as AV or not AV.
totally_unknown = {}

GOOD_MEDIA = {"audio", "video"}  # what you want to keep
bad, good = set(), set()  # buckets

# --- stage 1: MIME ----------------------------------------------------------
for ext in list(totally_unknown):
    mime, _ = mimetypes.guess_type("x." + ext)
    if mime and mime.split("/")[0] in GOOD_MEDIA:
        good.add(ext)
        totally_unknown.remove(ext)

# --- stage 2: ask ffmpeg ----------------------------------------------------
rx = re.compile(r"\b(audio|video|image|subtitle|data)\b", re.I)
for ext in list(totally_unknown):
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-h", f"muxer={ext}"], stderr=subprocess.STDOUT, text=True, timeout=2
        ).splitlines()[0]
        kind = rx.search(out).group(1).lower()
        (good if kind in GOOD_MEDIA else bad).add(ext)
    except Exception:
        bad.add(ext)

print("✓ audio/video:", sorted(good))
print("✗ not A/V    :", sorted(bad))
