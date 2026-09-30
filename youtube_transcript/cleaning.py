from __future__ import annotations

import re

from .models import TranscriptSegment

_BIDI_CONTROL_RE = re.compile("[‎‏‪-‮⁦-⁩]")

# Auto-captions wrap non-speech sounds in brackets, e.g. "[ضحك]" (laughter), "[أصوات شخير]" (snoring sounds), "[تنهّد]" (sigh) — these describe a sound, they aren't anyone speaking, so they don't belong in a dialogue transcript.
_BRACKET_RE = re.compile(r"\[[^\]]*\]")


class TranscriptCleaner:
    """Normalizes raw caption text and merges short fragments into readable lines."""

    def __init__(self, min_words: int = 6):
        self.min_words = min_words

    def clean(self, raw_segments) -> list[TranscriptSegment]:
        cleaned = [
            TranscriptSegment(
                text=self._remove_bracketed_annotations(self._strip_bidi_controls(segment.text)),
                start=segment.start,
                duration=segment.duration,
            )
            for segment in raw_segments
        ]
        return self._merge_short_segments(cleaned)

    @staticmethod
    def _strip_bidi_controls(text: str) -> str:
        return _BIDI_CONTROL_RE.sub("", text)

    @staticmethod
    def _remove_bracketed_annotations(text: str) -> str:
        text = _BRACKET_RE.sub("", text)
        return re.sub(r"\s+", " ", text).strip()

    def _merge_short_segments(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        merged: list[TranscriptSegment] = []
        for seg in segments:
            if len(seg.text.split()) < self.min_words and merged:
                previous = merged[-1]
                previous.duration = (seg.start + seg.duration) - previous.start
                if seg.text:
                    # ">>" marks a speaker change in YouTube auto-captions — put it on its
                    # own line within the merged cue instead of running it into the same line.
                    separator = "\n\n" if self._is_speaker_change(seg.text) else " "
                    previous.text = f"{previous.text}{separator}{seg.text}".strip()
            else:
                merged.append(TranscriptSegment(text=seg.text, start=seg.start, duration=seg.duration))
        return merged

    @staticmethod
    def _is_speaker_change(text: str) -> bool:
        return text.startswith(">>")
