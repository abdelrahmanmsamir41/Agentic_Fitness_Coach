from youtube_transcript import TranscriptCleaner, TranscriptError, TranscriptSegment, extract_video_id


def test_extract_video_id():
    assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s") == "dQw4w9WgXcQ"
    assert extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    try:
        extract_video_id("not a url")
        raise AssertionError("expected TranscriptError")
    except TranscriptError:
        pass


def test_cleaner_strips_brackets_and_merges_short_segments():
    raw = [
        TranscriptSegment(text="[laughter] hi", start=0.0, duration=1.0),
        TranscriptSegment(text="a fairly long segment here", start=1.0, duration=2.0),
    ]
    cleaned = TranscriptCleaner(min_words=6).clean(raw)
    assert len(cleaned) == 1
    assert cleaned[0].text == "hi a fairly long segment here"
    assert cleaned[0].duration == 3.0


def test_cleaner_merges_speaker_changes_onto_separate_lines_in_same_cue():
    raw = [
        TranscriptSegment(text=">> hi there", start=0.0, duration=1.0),
        TranscriptSegment(text=">> yeah sure", start=1.0, duration=1.0),
    ]
    cleaned = TranscriptCleaner(min_words=6).clean(raw)
    assert len(cleaned) == 1
    assert cleaned[0].text == ">> hi there\n\n>> yeah sure"
    assert cleaned[0].duration == 2.0


if __name__ == "__main__":
    test_extract_video_id()
    test_cleaner_strips_brackets_and_merges_short_segments()
    test_cleaner_merges_speaker_changes_onto_separate_lines_in_same_cue()
    print("ok")
