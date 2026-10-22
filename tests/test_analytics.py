from src.analytics import analytics_rows, summarize_detections


def test_counts_are_totaled_and_sorted_by_class():
    summary = summarize_detections([{"class_name": "person"}, {"class_name": "car"}, {"class_name": "person"}])
    assert summary == {"total": 3, "by_class": {"car": 1, "person": 2}}


def test_analytics_rows_are_csv_friendly():
    rows = analytics_rows([{"frame": 4, "time_seconds": 0.5, "inference_ms": 11,
                            "detections": [{"class_name": "car"}, {"class_name": "car"}]}])
    assert rows == [{"frame": 4, "time_seconds": 0.5, "inference_ms": 11,
                     "total_objects": 2, "class_counts": "car:2"}]
