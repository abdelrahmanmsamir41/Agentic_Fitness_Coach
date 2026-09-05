from __future__ import annotations

import sys

from .exceptions import TranscriptError
from .fetcher import TranscriptFetcher
from .formatters import OUTPUT_FORMATS
from .video_id import extract_video_id


def main() -> None:
    if len(sys.argv) not in (2, 3):
        print("Usage: python -m youtube_transcript <youtube_url_or_video_id> [text|srt|vtt]")
        raise SystemExit(1)

    fmt = sys.argv[2] if len(sys.argv) == 3 else "text"
    if fmt not in OUTPUT_FORMATS:
        print(f"Unknown format {fmt!r} — choose one of: {', '.join(OUTPUT_FORMATS)}")
        raise SystemExit(1)

    try:
        video_id = extract_video_id(sys.argv[1])
        segments = TranscriptFetcher().fetch(video_id, ["en"])

        formatter, extension = OUTPUT_FORMATS[fmt]
        output_file = f"transcript_{video_id}.{extension}"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(formatter(segments))

        print(f"Transcript saved to '{output_file}'")
    except TranscriptError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
