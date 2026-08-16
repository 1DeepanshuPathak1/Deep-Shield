import json
import os
import re
import subprocess
from urllib.parse import urlparse

BROWSER_SAFE_VIDEO = {"h264", "avc1", "vp8", "vp9", "av1"}
BROWSER_SAFE_AUDIO = {"aac", "mp3", "opus", "vorbis", "flac"}

ALLOWED_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
}

CLIP_SECONDS = 30


class MediaError(Exception):
    pass


def _run(command, timeout=300):
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return result


def probe(path):
    result = _run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            path,
        ],
        timeout=60,
    )
    if result.returncode != 0:
        raise MediaError("The media file could not be inspected.")

    data = json.loads(result.stdout.decode("utf-8", "replace") or "{}")
    video = None
    audio = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and video is None:
            video = stream
        elif stream.get("codec_type") == "audio" and audio is None:
            audio = stream

    duration = data.get("format", {}).get("duration")
    return {
        "videoCodec": (video or {}).get("codec_name"),
        "audioCodec": (audio or {}).get("codec_name"),
        "hasVideo": video is not None,
        "hasAudio": audio is not None,
        "width": (video or {}).get("width"),
        "height": (video or {}).get("height"),
        "duration": round(float(duration), 2) if duration else None,
    }


def browser_playable(info):
    if not info["hasVideo"]:
        return False
    if (info["videoCodec"] or "").lower() not in BROWSER_SAFE_VIDEO:
        return False
    if info["hasAudio"] and (info["audioCodec"] or "").lower() not in BROWSER_SAFE_AUDIO:
        return False
    return True


def transcode_to_h264(source, destination, max_height=720):
    result = _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            source,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "26",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            f"scale=-2:'min({max_height},ih)'",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            destination,
        ],
        timeout=900,
    )
    if result.returncode != 0 or not os.path.exists(destination):
        raise MediaError("The video could not be converted for playback.")
    return destination


def extract_audio(source, destination, sample_rate=16000):
    result = _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            source,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            destination,
        ],
        timeout=600,
    )
    if result.returncode != 0 or not os.path.exists(destination):
        raise MediaError("No audio track could be extracted from this file.")
    return destination


def validate_youtube_url(url):
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        raise MediaError("Enter a full YouTube link starting with https://")
    if parsed.hostname not in ALLOWED_HOSTS:
        raise MediaError("Only YouTube links are supported.")
    if not re.search(r"[A-Za-z0-9_-]{6,}", parsed.path + (parsed.query or "")):
        raise MediaError("That link does not contain a video id.")
    return url.strip()


def describe(url):
    import yt_dlp

    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise MediaError(f"That link could not be read: {str(exc)[:160]}")

    if not info:
        raise MediaError("That video could not be reached.")

    return {
        "title": info.get("title") or "Untitled",
        "channel": info.get("uploader") or info.get("channel") or "Unknown",
        "isLive": bool(info.get("is_live")),
        "duration": info.get("duration"),
        "thumbnail": info.get("thumbnail"),
    }


def _live_manifest(url):
    import yt_dlp

    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "format": "best[height<=720]/best",
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)

    stream = info.get("manifest_url") or info.get("url")
    if not stream:
        for candidate in info.get("formats", []) or []:
            if candidate.get("protocol", "").startswith("m3u8") and candidate.get("url"):
                stream = candidate["url"]
                break
    if not stream:
        raise MediaError("No live stream could be resolved for that link.")
    return stream


def _ffmpeg_capture(stream, destination, seconds):
    result = _run(
        [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-i",
            stream,
            "-t",
            str(seconds),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "26",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "scale=-2:'min(480,ih)'",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            destination,
        ],
        timeout=600,
    )
    if result.returncode != 0 or not os.path.exists(destination):
        detail = result.stderr.decode("utf-8", "replace").strip().splitlines()
        tail = detail[-1] if detail else "unknown error"
        raise MediaError(f"The live clip could not be captured: {tail[:160]}")
    return destination


def capture_clip(url, destination, seconds=CLIP_SECONDS, live=False):
    import yt_dlp

    stem = destination[:-4] if destination.endswith(".mp4") else destination

    if live:
        stream = _live_manifest(url)
        return _ffmpeg_capture(stream, destination, seconds)

    source_stem = stem + "_src"
    options = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "outtmpl": source_stem + ".%(ext)s",
        "format": (
            "best[height<=480][ext=mp4]/best[height<=480]/"
            "best[height<=720][ext=mp4]/best[ext=mp4]/best"
        ),
        "overwrites": True,
        "retries": 5,
        "fragment_retries": 5,
        "extractor_args": {"youtube": {"player_client": ["android", "web", "tv"]}},
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        },
    }

    last_error = None
    for client in (["android"], ["web"], ["tv"], ["ios"]):
        options["extractor_args"] = {"youtube": {"player_client": client}}
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([url])
            last_error = None
            break
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise MediaError(
            f"The clip could not be downloaded. YouTube refused the request "
            f"({str(last_error)[:120]}). This is usually rate limiting; try again shortly."
        )

    folder = os.path.dirname(source_stem) or "."
    base = os.path.basename(source_stem)
    candidates = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.startswith(base) and not name.endswith((".part", ".ytdl", ".temp"))
    ]
    if not candidates:
        raise MediaError("The download produced no file.")

    produced = max(candidates, key=os.path.getsize)

    result = _run(
        [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-i",
            produced,
            "-t",
            str(seconds),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "26",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "scale=-2:'min(480,ih)'",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            destination,
        ],
        timeout=600,
    )
    try:
        os.remove(produced)
    except OSError:
        pass

    if result.returncode != 0 or not os.path.exists(destination):
        raise MediaError("The captured clip could not be prepared for analysis.")

    return destination
