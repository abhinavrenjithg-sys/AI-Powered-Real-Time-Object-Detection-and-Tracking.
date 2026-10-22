"""Input validation helpers for user-supplied media."""
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"}


def validate_media_name(name: str, kind: str) -> tuple[bool, str]:
    """Validate a filename extension; decoding is performed by OpenCV."""
    suffix = Path(name).suffix.lower()
    supported = IMAGE_SUFFIXES if kind == "image" else VIDEO_SUFFIXES if kind == "video" else set()
    if not supported:
        return False, f"Unsupported media type: {kind}"
    if suffix not in supported:
        return False, f"Unsupported {kind} format: {suffix or '(no extension)'}"
    return True, ""
