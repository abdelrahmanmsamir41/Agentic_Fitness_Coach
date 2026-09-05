from __future__ import annotations

from youtube_transcript_api.formatters import SRTFormatter, WebVTTFormatter

from .models import TranscriptSegment


def format_timestamp(seconds: float) -> str:
    """MM:SS, or HH:MM:SS once the video runs an hour or longer."""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def export_txt(segments: list[TranscriptSegment]) -> str:
    return "\n".join(f"[{format_timestamp(s.start)}] {s.text}" for s in segments)


def export_srt(segments: list[TranscriptSegment]) -> str:
    """Format the transcript as a standard .srt subtitle file."""
    return SRTFormatter().format_transcript(segments)


def export_vtt(segments: list[TranscriptSegment]) -> str:
    """Format the transcript as a standard WebVTT (.vtt) subtitle file."""
    return WebVTTFormatter().format_transcript(segments)


OUTPUT_FORMATS = {
    "text": (export_txt, "txt"),
    "srt": (export_srt, "srt"),
    "vtt": (export_vtt, "vtt"),
}
