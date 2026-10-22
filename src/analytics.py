"""Pure analytics helpers, kept independent of the model runtime."""
from collections import Counter
from typing import Iterable, Mapping, Any


def summarize_detections(detections: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return total and per-class counts from normalized detection records."""
    records = list(detections)
    counts = Counter(str(item.get("class_name", "unknown")) for item in records)
    return {"total": len(records), "by_class": dict(sorted(counts.items()))}


def analytics_rows(frame_records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flatten frame-level records into CSV-friendly rows."""
    rows = []
    for frame in frame_records:
        counts = summarize_detections(frame.get("detections", []))
        rows.append({"frame": frame.get("frame", 0), "time_seconds": frame.get("time_seconds", 0.0),
                     "inference_ms": frame.get("inference_ms", 0.0), "total_objects": counts["total"],
                     "class_counts": ";".join(f"{k}:{v}" for k, v in counts["by_class"].items())})
    return rows
