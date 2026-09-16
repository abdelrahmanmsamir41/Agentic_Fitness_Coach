"""Pull everything yt-dlp can get from a YouTube URL besides the transcript."""
import argparse
import json
from pathlib import Path

import yt_dlp


def extract(url: str, out_dir: Path, max_comments: int = 20) -> dict:
    """Metadata + comments (no file download involved, so a rate limit elsewhere can't break it)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "skip_download": True,
        "noplaylist": True,
        "getcomments": True,
        "extractor_args": {"youtube": {"max_comments": [str(max_comments)]}},
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(url, download=False)


def download_files(url: str, out_dir: Path) -> dict:
    """Best-effort thumbnail download — failures here don't affect metadata above."""
    ydl_opts = {
        "skip_download": True,
        "noplaylist": True,
        "ignoreerrors": "only_download",
        "writethumbnail": True,
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "quiet": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return {"ok": True}
    except yt_dlp.utils.DownloadError as exc:
        return {"ok": False, "error": str(exc)}


def summarize(info: dict) -> dict:
    return {
        "id": info.get("id"),
        "title": info.get("title"),
        "duration_s": info.get("duration"),
        "upload_date": info.get("upload_date"),
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
        "channel": info.get("channel"),
        "tags": info.get("tags"),
        "categories": info.get("categories"),
        "num_formats": len(info.get("formats") or []),
        "chapters": info.get("chapters"),
        "num_comments_fetched": len(info.get("comments") or []),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url", nargs="?", default="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    parser.add_argument("--out", default=r"ytdlp-test\output")
    parser.add_argument("--max-comments", type=int, default=20)
    args = parser.parse_args()

    info = extract(args.url, Path(args.out), args.max_comments)
    with open(Path(args.out) / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"Wrote metadata.json to {args.out}")
    summary = summarize(info)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    file_result = download_files(args.url, Path(args.out))
    if file_result["ok"]:
        print("\nThumbnail downloaded to", args.out)
    else:
        print("\nThumbnail download failed (metadata above is unaffected):", file_result["error"])

    # self-check: the fields this script promises must actually be present
    assert summary["id"], "no video id extracted"
    assert summary["num_formats"] > 0, "no formats extracted"
    print("\nOK — metadata and formats extracted successfully.")
