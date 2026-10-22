from src.detector import normalize_result


class Scalar:
    def __init__(self, value): self.value = value
    def item(self): return self.value


class Box:
    xyxy = [[1, 2, 30, 40]]
    cls = [Scalar(2)]
    conf = [Scalar(0.875)]
    id = [Scalar(7)]


class Result:
    boxes = [Box()]
    names = {2: "car"}


def test_normalize_detection_and_tracking_output():
    assert normalize_result(Result()) == [{"xyxy": [1., 2., 30., 40.], "class_id": 2,
        "class_name": "car", "confidence": 0.875, "track_id": 7}]


def test_empty_or_missing_boxes_produce_empty_list():
    assert normalize_result(None) == []
    assert normalize_result(type("Empty", (), {"boxes": None})()) == []
