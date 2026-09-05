from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

from .exceptions import TranscriptError

_VIDEO_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{11}$")


def extract_video_id(url_or_id: str) -> str:
    candidate = url_or_id.strip()

    if _VIDEO_ID_RE.match(candidate):
        return candidate

    parsed = urlparse(candidate)
    host = (parsed.netloc or "").lower()

    video_id = ""
    if "youtu.be" in host:
        video_id = parsed.path.lstrip("/")
    elif "youtube.com" in host:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            for prefix in ("/embed/", "/shorts/", "/live/"):
                if parsed.path.startswith(prefix):
                    video_id = parsed.path[len(prefix):]
                    break

    video_id = video_id.split("&", 1)[0].split("?", 1)[0]

    if not _VIDEO_ID_RE.match(video_id):
        raise TranscriptError(f"Could not extract a valid video ID from: {url_or_id!r}")

    return video_id
