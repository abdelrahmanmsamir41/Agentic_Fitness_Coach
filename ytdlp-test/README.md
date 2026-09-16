# yt-dlp feature test

yt-dlp was pinned at 2024.4.9 (broken signature extraction against current YouTube).
Updated to 2026.8.19 — install into your venv with `pip install yt-dlp`.

Transcripts are fetched exclusively via the `youtube_transcript` package now
(see its own README) — yt-dlp here only pulls what that package doesn't:
metadata, formats, tags, chapters, comments, thumbnail.

## Scripts

| Script | What it does | Output folder |
|---|---|---|
| `extract_youtube_features.py URL [--out DIR]` | Metadata, formats, tags, chapters, comments; best-effort thumbnail file | `output/` (default) |

`--out` controls where files land — nothing writes loose into this folder
itself. The default is gitignored scratch output, safe to delete anytime.

## Known gotchas found while testing

- A URL with `&list=...&index=...` (playlist-linked) makes yt-dlp resolve the whole
  playlist unless `noplaylist: True` is set — this script sets it.
- Thumbnail download can fail independently of metadata — `extract_youtube_features.py`
  fetches metadata with `download=False` first so that can't take metadata down with it.
