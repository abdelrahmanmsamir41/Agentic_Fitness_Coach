from __future__ import annotations

from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    IpBlocked,
    NoTranscriptFound,
    NotTranslatable,
    RequestBlocked,
    TranscriptsDisabled,
    TranslationLanguageNotAvailable,
    VideoUnavailable,
    YouTubeTranscriptApi,
)
from youtube_transcript_api.proxies import ProxyConfig

from .cleaning import TranscriptCleaner
from .exceptions import TranscriptError
from .models import TranscriptSegment
from .video_id import extract_video_id


class TranscriptFetcher:
    """Fetches a YouTube transcript, falling back to translation/any-available-language, and cleans it."""

    def __init__(self, proxy_config: ProxyConfig | None = None, cleaner: TranscriptCleaner | None = None):
        self.api = YouTubeTranscriptApi(proxy_config=proxy_config)
        self.cleaner = cleaner or TranscriptCleaner()

    def fetch(self, url_or_id: str, languages: list[str] | None = None) -> list[TranscriptSegment]:
        video_id = extract_video_id(url_or_id)
        languages = languages or ["en"]

        try:
            fetched = self._fetch_with_fallback(video_id, languages)
        except TranscriptsDisabled as exc:
            raise TranscriptError(f"Transcripts are disabled for video {video_id}") from exc
        except VideoUnavailable as exc:
            raise TranscriptError(f"Video {video_id} is unavailable") from exc
        except (IpBlocked, RequestBlocked) as exc:
            raise TranscriptError(
                f"YouTube blocked this request for video {video_id} (common on cloud/"
                "datacenter IPs). Retry from a residential connection, or pass a "
                "proxy_config= (see youtube_transcript_api.proxies)."
            ) from exc
        except CouldNotRetrieveTranscript as exc:
            raise TranscriptError(f"Could not retrieve transcript for {video_id}: {exc}") from exc

        return self.cleaner.clean(fetched)

    def _fetch_with_fallback(self, video_id: str, languages: list[str]):
        try:
            return self.api.fetch(video_id, languages=languages)
        except NoTranscriptFound:
            transcript_list = self.api.list(video_id)

            for lang in languages:
                for transcript in transcript_list:
                    if not transcript.is_translatable:
                        continue
                    try:
                        return transcript.translate(lang).fetch()
                    except (NotTranslatable, TranslationLanguageNotAvailable):
                        continue

            # Still nothing in a preferred language, natively or translated — fall back to whatever the video actually has (e.g. auto-generated, original language).
            try:
                transcript = next(iter(transcript_list))
            except StopIteration as exc:
                raise TranscriptError(f"No transcript available for video {video_id}") from exc
            return transcript.fetch()


def get_transcript(
    url_or_id: str,
    languages: list[str] | None = None,
    proxy_config: ProxyConfig | None = None,
) -> list[TranscriptSegment]:
    """Functional convenience wrapper around TranscriptFetcher for one-off calls."""
    return TranscriptFetcher(proxy_config=proxy_config).fetch(url_or_id, languages)
