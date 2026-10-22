"""Video processing and annotation utilities."""
import time
import cv2


def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["xyxy"])
        label = f"{det['class_name']} {det['confidence']:.2f}"
        if det.get("track_id") is not None:
            label += f" ID:{det['track_id']}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (48, 190, 120), 2)
        cv2.putText(frame, label, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (48, 190, 120), 2)
    return frame


def process_video(input_path, output_path, detector, confidence=0.25, classes=None, line_y=None):
    """Process a video through YOLO tracking, saving annotated output and records."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError("OpenCV could not open this video. Check its encoding and file contents.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Video has invalid dimensions or an unsupported codec.")
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create the output video on this system.")
    records, crossing_ids, crossings, frame_no = [], set(), 0, 0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            tick = time.perf_counter()
            dets = detector.track(frame, confidence=confidence, classes=classes)
            inference_ms = (time.perf_counter() - tick) * 1000
            if line_y is not None:
                for det in dets:
                    tid = det.get("track_id")
                    center_y = (det["xyxy"][1] + det["xyxy"][3]) / 2
                    if tid is not None and center_y >= line_y and tid not in crossing_ids:
                        crossing_ids.add(tid)
                        crossings += 1
                cv2.line(frame, (0, int(line_y)), (width, int(line_y)), (0, 200, 255), 2)
            draw_detections(frame, dets)
            writer.write(frame)
            records.append({"frame": frame_no, "time_seconds": frame_no / fps, "inference_ms": inference_ms,
                            "detections": dets})
            frame_no += 1
    finally:
        cap.release()
        writer.release()
    return {"records": records, "duration_seconds": time.perf_counter() - started,
            "fps": fps, "frames": frame_no, "line_crossings": crossings}
