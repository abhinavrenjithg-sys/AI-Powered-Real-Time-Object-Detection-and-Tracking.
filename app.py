"""Streamlit dashboard for AI-powered object detection and tracking."""
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.analytics import analytics_rows, summarize_detections
from src.detector import ModelUnavailableError, ObjectDetector
from src.validation import validate_media_name
from src.video import draw_detections, process_video

st.set_page_config(page_title="VisionTrack AI", page_icon="🎯", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root{--ink:#edf3ff;--muted:#9caac2;--line:rgba(148,173,211,.16);--mint:#62e6bd;--blue:#91a8ff}
.stApp{background:radial-gradient(ellipse at 84% -10%,rgba(69,104,169,.23),transparent 42%),linear-gradient(145deg,#0a101c 0%,#0d1423 56%,#101a2c 100%);color:var(--ink);font-family:'DM Sans',sans-serif}
.block-container{max-width:1440px;padding-top:2.1rem;padding-bottom:3rem}
[data-testid="stSidebar"]>div{background:linear-gradient(180deg,#101a2d 0%,#0b1220 100%);border-right:1px solid var(--line)}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1{font-family:'Manrope',sans-serif;letter-spacing:-.04em}
[data-testid="stSidebar"] hr{border-color:var(--line)}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p{color:#b6c3d8}
[data-testid="stRadio"] label{padding:.18rem .25rem;border-radius:8px}
[data-testid="stRadio"] label:hover{background:rgba(119,151,210,.1)}
.hero{position:relative;overflow:hidden;padding:1.9rem 2rem;margin-bottom:1.2rem;border:1px solid rgba(144,174,224,.2);border-radius:22px;background:linear-gradient(118deg,rgba(30,47,77,.94),rgba(16,27,47,.94) 60%,rgba(23,39,64,.9));box-shadow:0 20px 60px rgba(0,0,0,.18)}
.hero:after{content:'';position:absolute;right:-85px;top:-155px;width:330px;height:330px;border-radius:50%;background:radial-gradient(circle,rgba(98,230,189,.19),rgba(111,145,255,.08) 45%,transparent 72%);pointer-events:none}
.hero h1{font-family:'Manrope',sans-serif;font-size:clamp(2rem,4vw,2.65rem);line-height:1.1;letter-spacing:-.05em;margin:0 0 .55rem;color:#f2f6ff}
.hero p{color:#b2c2dc;font-size:1.02rem;margin:0;max-width:680px}
.hero .eyebrow{display:inline-flex;align-items:center;gap:.45rem;color:var(--mint);font-size:.72rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase;margin-bottom:.8rem}
div[data-testid="stMetric"]{background:linear-gradient(145deg,rgba(22,34,56,.96),rgba(17,27,45,.92));padding:1.1rem 1.2rem;border-radius:16px;border:1px solid var(--line);box-shadow:0 10px 28px rgba(0,0,0,.12)}
div[data-testid="stMetric"] label{color:#aab9d0} div[data-testid="stMetric"] [data-testid="stMetricValue"]{font-family:'Manrope',sans-serif;letter-spacing:-.04em}
h1,h2,h3{font-family:'Manrope',sans-serif;letter-spacing:-.035em}h2{margin-top:1.7rem}
button[kind="primary"]{background:linear-gradient(110deg,#48d6b0,#7ce6c7);color:#081b1b;border:0;border-radius:10px;font-weight:700;box-shadow:0 8px 22px rgba(64,217,177,.18)}
button[kind="primary"]:hover{background:linear-gradient(110deg,#74e7c8,#a1f2db);color:#081b1b;border:0}
button[kind="secondary"]{border-radius:10px;border-color:var(--line)}
[data-testid="stFileUploader"] section{border:1px dashed rgba(145,168,255,.42);border-radius:16px;background:rgba(18,29,49,.6)}
[data-testid="stFileUploader"] section:hover{border-color:var(--mint)}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:14px;overflow:hidden}
[data-testid="stAlert"]{border-radius:12px}
.stCaption{color:var(--muted)}
</style>""", unsafe_allow_html=True)

PAGES = ["Overview", "Image Detection", "Video Detection", "Live Camera", "Analytics", "About"]
CLASS_OPTIONS = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
COCO_IDS = {name: i for i, name in enumerate(CLASS_OPTIONS)}

with st.sidebar:
    st.markdown("# 🎯 VisionTrack")
    page = st.radio("Workspace", PAGES, label_visibility="collapsed")
    st.divider()
    model_path = st.text_input("Model weights", os.getenv("YOLO_MODEL_PATH", "models/yolo11n.pt"), help="Local Ultralytics compatible .pt file")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    selected = st.multiselect("Classes", CLASS_OPTIONS, default=["person", "car", "truck", "bus"])
    class_ids = [COCO_IDS[x] for x in selected] if selected else None

@st.cache_resource
def detector_for(path):
    return ObjectDetector(path)

def get_detector():
    try:
        return detector_for(model_path)
    except Exception as e:
        st.error(f"Could not initialize detector: {e}")
        return None

def hero(title, subtitle):
    st.markdown(f'<div class="hero"><div class="eyebrow">✦ Vision intelligence workspace</div><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

def model_error(error):
    st.error(str(error))
    st.info("Place compatible local weights in models/ or set YOLO_MODEL_PATH. The app does not download weights automatically.")

if page == "Overview":
    hero("AI-Powered Object Detection", "Real-time vision analytics, tracking and media processing in one workspace.")
    st.write("")
    cols = st.columns(4)
    for col, (value, label) in zip(cols, [("YOLO", "Detection engine"), ("ByteTrack", "Persistent IDs"), ("6", "Dashboard views"), ("Local", "Media processing")]):
        col.metric(label, value)
    st.subheader("Start exploring")
    st.write("Upload an image or video, or connect a local webcam. Tune confidence and object classes from the sidebar.")
    st.caption("Model weights are loaded from your configured local path; no automatic large downloads.")
elif page == "Image Detection":
    hero("Image Detection", "Upload a still image and inspect model predictions.")
    upload = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    sample_clicked = st.button("Run included sample", type="secondary")
    if not upload:
        st.caption("No test image? Try the included, attributed sample photo.")
    if upload or sample_clicked:
        if upload:
            ok, why = validate_media_name(upload.name, "image")
        else:
            ok, why = True, ""
        if not ok: st.error(why)
        else:
            if upload:
                raw = upload.getvalue()
                image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            else:
                image = cv2.imread("data/samples/sample_bus_staff.jpg")
                st.caption("Sample image: LR.127, Wikimedia Commons, CC BY 4.0 (credit details in data/samples/ATTRIBUTION.md).")
            if image is None: st.error("This image could not be decoded.")
            else:
                detector = get_detector()
                if detector:
                    try:
                        started = time.perf_counter(); detections = detector.detect(image, confidence, class_ids); elapsed = time.perf_counter()-started
                        st.image(cv2.cvtColor(draw_detections(image.copy(), detections), cv2.COLOR_BGR2RGB), use_container_width=True)
                        a,b,c=st.columns(3); a.metric("Objects",len(detections)); b.metric("Inference",f"{elapsed*1000:.1f} ms"); c.metric("Processing",f"{elapsed:.2f} s")
                        st.dataframe(pd.DataFrame(detections), use_container_width=True)
                    except ModelUnavailableError as e: model_error(e)
elif page == "Video Detection":
    hero("Video Detection", "Track objects through uploaded video frames and export annotated media.")
    upload = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv", "mpeg", "mpg"])
    crossing = st.checkbox("Count objects crossing a horizontal line")
    if upload:
        valid, why = validate_media_name(upload.name, "video")
        if not valid: st.error(why)
        elif st.button("Process video", type="primary"):
            detector = get_detector()
            if detector:
                with tempfile.TemporaryDirectory() as td:
                    in_path=Path(td)/Path(upload.name).name; out_path=Path(td)/"processed.mp4"; in_path.write_bytes(upload.getvalue())
                    cap=cv2.VideoCapture(str(in_path)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
                    try:
                        with st.spinner("Processing video…"):
                            result=process_video(in_path,out_path,detector,confidence,class_ids,h//2 if crossing else None)
                        st.success(f"Processed {result['frames']} frames in {result['duration_seconds']:.2f} s")
                        st.video(out_path.read_bytes()); st.download_button("Download processed video",out_path.read_bytes(),"processed.mp4","video/mp4")
                        st.metric("Line crossings",result["line_crossings"])
                        rows=analytics_rows(result["records"]); csv=pd.DataFrame(rows).to_csv(index=False)
                        st.download_button("Download analytics CSV",csv,"analytics.csv","text/csv")
                        st.session_state["last_analytics"]=rows
                    except (ModelUnavailableError, ValueError, OSError) as e: model_error(e)
elif page == "Live Camera":
    hero("Live Camera", "Run tracking from a camera available to the machine hosting Streamlit.")
    camera_index=st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)
    frames=st.number_input("Frames to capture", min_value=1, max_value=500, value=120, step=10)
    if st.button("Start camera", type="primary"):
        detector=get_detector()
        if detector:
            cap=cv2.VideoCapture(int(camera_index))
            if not cap.isOpened(): st.error("Camera unavailable. Check permissions, index, and whether another app is using it.")
            else:
                placeholder=st.empty(); records=[]; start=time.perf_counter()
                try:
                    for frame_no in range(int(frames)):
                        ok,frame=cap.read()
                        if not ok: st.warning("Camera stopped returning frames."); break
                        tick=time.perf_counter()
                        try: detections=detector.track(frame,confidence,class_ids)
                        except ModelUnavailableError as e: model_error(e); break
                        infer=(time.perf_counter()-tick)*1000; records.append({"frame":frame_no,"time_seconds":frame_no/30,"inference_ms":infer,"detections":detections})
                        placeholder.image(cv2.cvtColor(draw_detections(frame,detections),cv2.COLOR_BGR2RGB),use_container_width=True)
                    st.success(f"Processed {len(records)} frames in {time.perf_counter()-start:.2f} s")
                    st.download_button("Download camera analytics CSV",pd.DataFrame(analytics_rows(records)).to_csv(index=False),"camera_analytics.csv","text/csv")
                    st.session_state["last_analytics"]=analytics_rows(records)
                finally: cap.release()
elif page == "Analytics":
    hero("Analytics", "Review the most recent media run and export frame-level counts.")
    rows=st.session_state.get("last_analytics",[])
    if rows:
        df=pd.DataFrame(rows); a,b=st.columns(2); a.metric("Frames",len(df)); b.metric("Average inference",f"{df['inference_ms'].mean():.1f} ms")
        st.line_chart(df.set_index("time_seconds")[["total_objects"]]); st.dataframe(df,use_container_width=True)
        st.download_button("Export CSV",df.to_csv(index=False),"analytics.csv","text/csv")
    else: st.info("Process a video or camera session to populate analytics.")
else:
    hero("About VisionTrack", "A portfolio project demonstrating an end-to-end computer vision workflow.")
    st.markdown("""### Technical choices
- **Ultralytics YOLO + PyTorch:** compact pretrained detector interface with configurable local weights.
- **ByteTrack:** tracking-by-detection associates observations over time and supplies temporary track IDs.
- **OpenCV:** decoding, drawing, camera capture and video writing.
- **Streamlit:** rapid interactive dashboard without a separate frontend stack.

### Hardware and performance
CPU inference works but may be slow; a supported CUDA GPU can improve throughput. Actual latency depends on model size, input resolution, hardware, video codec and concurrent load. The UI reports measurements from your run; no benchmark is claimed here.

### Limitations, licensing and responsible use
This is an educational portfolio prototype, not a validated production system. Detection quality varies with domain, lighting, occlusion and camera angle. Tracking IDs can switch or fragment. Line crossing currently counts each ID once when its center passes the configured horizontal threshold; it does not infer travel direction or robustly deduplicate reappearing IDs. Verify the chosen weights, framework, and any datasets for their licenses before redistribution or commercial use. Use only with appropriate consent, privacy controls and legal basis. Do not use it to identify people or make high-impact decisions.
""")
