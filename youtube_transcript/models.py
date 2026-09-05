from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TranscriptSegment:
    text: str
    start: float      # seconds from the start of the video
    duration: float    # how long this segment is shown, in seconds
