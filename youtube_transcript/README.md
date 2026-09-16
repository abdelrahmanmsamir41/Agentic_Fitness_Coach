# youtube_transcript

Fetches, cleans, and exports YouTube transcripts via `youtube_transcript_api` — the
project's primary transcript source (see `ytdlp-test/` for yt-dlp, used for
everything else: metadata, formats, chapters, comments).

## Usage

```
python -m youtube_transcript <youtube_url_or_video_id> [text|srt|vtt]
python -m youtube_transcript <youtube_url_or_video_id> --list-langs
```

Saves `transcript_<video_id>.<ext>` in the current directory. `--list-langs` instead
prints `list_languages()`'s result as JSON.

Programmatically:

```python
from youtube_transcript import get_transcript, list_languages, export_srt

segments = get_transcript("https://www.youtube.com/watch?v=...", languages=["en"])
list_languages("...")  # {"manual": [...], "auto_generated": [...], "translatable_to": [...]}
export_srt(segments)
```

## Modules

| File | Responsibility |
|---|---|
| `video_id.py` | Extracts a bare video ID from any YouTube URL form (watch, youtu.be, embed, shorts, live, playlist-linked) |
| `fetcher.py` | `TranscriptFetcher` — fetches a transcript, falling back manual → translated → whatever's available |
| `languages.py` | `list_languages()` — manual vs. native auto-generated languages, plus what the auto track can actually be translated into (per YouTube's own metadata, not yt-dlp's speculative full menu) |
| `cleaning.py` | `TranscriptCleaner` — strips bracketed sound annotations (`[music]`) and bidi control characters, merges short fragments into readable lines |
| `formatters.py` | Exports cleaned segments as text/srt/vtt |
| `models.py` | `TranscriptSegment` dataclass (text, start, duration) |
| `exceptions.py` | `TranscriptError` — the one exception type callers need to catch |

## Known limitations

- Can get `IpBlocked`/`RequestBlocked` on cloud/datacenter IPs (per `fetcher.py`) —
  different failure mode from yt-dlp's 429s, but still a real one. Pass a
  `proxy_config=` to `TranscriptFetcher`/`get_transcript` if that happens.
- `list_languages()`'s `translatable_to` reflects what YouTube actually advertises for
  that video — it's usually much smaller (~15-20 languages) than yt-dlp's
  `automatic_captions` count (100+), because yt-dlp lists its full UI translate menu
  unconditionally rather than per-video capability. Treat this library's number as
  the trustworthy one.
