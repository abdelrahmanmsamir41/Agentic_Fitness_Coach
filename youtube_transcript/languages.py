from __future__ import annotations

from youtube_transcript_api import YouTubeTranscriptApi

from .video_id import extract_video_id


def list_languages(url_or_id: str) -> dict:
    """Manual vs. native auto-generated caption languages, plus languages the auto track can be translated into."""
    video_id = extract_video_id(url_or_id)
    transcript_list = YouTubeTranscriptApi().list(video_id)

    manual, auto_generated, translatable_to = [], [], set()
    for transcript in transcript_list:
        (auto_generated if transcript.is_generated else manual).append(transcript.language_code)
        if transcript.is_translatable:
            translatable_to.update(language.language_code for language in transcript.translation_languages)

    return {
        "manual": sorted(manual),
        "auto_generated": sorted(auto_generated),
        "translatable_to": sorted(translatable_to),
    }
