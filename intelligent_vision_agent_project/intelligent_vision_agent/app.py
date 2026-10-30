import time
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO

st.set_page_config(page_title="Intelligent Vision Agent", page_icon="👁️", layout="wide")

st.markdown("""
<style>
.main {background: #f7f8fc;}
[data-testid="stSidebar"] {background: #111827;}
[data-testid="stSidebar"] * {color: #f9fafb;}
.hero {padding: 1.2rem 1.4rem; border-radius: 18px; color: white;
       background: linear-gradient(120deg,#101827,#243b6b,#7c3aed); margin-bottom: 1rem;}
.hero h1 {margin: 0; font-size: 2rem;}
.hero p {margin: .4rem 0 0 0; color: #dbeafe;}
.metric {background:white; border:1px solid #e5e7eb; padding:1rem; border-radius:14px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

st.sidebar.title("👁️ Intelligent Vision")
page = st.sidebar.radio("Workspace", [
    "Image Analysis", "Detection History", "AI Performance", "System Status", "About Project"
])
st.sidebar.caption("Computer vision • Object detection • Visual insights")
model_name = st.sidebar.selectbox("Detection model", ["yolov8n.pt", "yolov8s.pt"], index=0)
confidence = st.sidebar.slider("Confidence threshold", 0.10, 0.90, 0.35, 0.05)
st.sidebar.info("First run downloads the selected pretrained YOLO model. An internet connection may be required.")

if "history" not in st.session_state:
    st.session_state.history = []

def run_detection(image, model_name, conf):
    model = load_model(model_name)
    start = time.perf_counter()
    result = model.predict(image, conf=conf, verbose=False)[0]
    elapsed = (time.perf_counter() - start) * 1000
    annotated = result.plot()
    annotated_image = Image.fromarray(annotated[:, :, ::-1])
    rows = []
    names = result.names
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            rows.append({
                "Object": names.get(cls_id, str(cls_id)),
                "Confidence": round(float(box.conf.item()), 4),
                "X1": round(float(box.xyxy[0][0].item()), 1),
                "Y1": round(float(box.xyxy[0][1].item()), 1),
                "X2": round(float(box.xyxy[0][2].item()), 1),
                "Y2": round(float(box.xyxy[0][3].item()), 1),
            })
    return annotated_image, pd.DataFrame(rows), elapsed

def summary_text(df):
    if df.empty:
        return "No objects were detected at the selected confidence threshold. Try lowering the threshold or uploading a clearer image."
    counts = df["Object"].value_counts()
    most_common = counts.index[0]
    return (f"The system detected **{len(df)} object(s)** across **{df['Object'].nunique()} class(es)**. "
            f"The most frequent class was **{most_common}** ({int(counts.iloc[0])} detection(s)). "
            f"Average detection confidence: **{df['Confidence'].mean():.1%}**.")

st.markdown('<div class="hero"><h1>Intelligent Vision Agent</h1><p>AI-powered image understanding, object detection, and visual analysis</p></div>', unsafe_allow_html=True)

if page == "Image Analysis":
    st.write("Upload an image to detect objects, inspect confidence scores, and review a concise visual summary.")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        left, right = st.columns(2)
        with left:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Running the vision pipeline…"):
                try:
                    annotated, df, elapsed = run_detection(image, model_name, confidence)
                    st.session_state.current_result = {
                        "filename": uploaded.name, "annotated": annotated, "df": df,
                        "elapsed": elapsed, "image": image.copy()
                    }
                    st.session_state.history.append({
                        "Image": uploaded.name, "Objects": len(df),
                        "Classes": int(df["Object"].nunique()) if not df.empty else 0,
                        "Time (ms)": round(elapsed, 1),
                        "Avg confidence": round(float(df["Confidence"].mean()), 3) if not df.empty else 0.0
                    })
                except Exception as e:
                    st.error(f"Could not run detection: {e}")
                    st.info("Check your internet connection for the first model download, then try again.")
        current = st.session_state.get("current_result")
        if current and current["filename"] == uploaded.name:
            with right:
                st.subheader("Detected Objects")
                st.image(current["annotated"], use_container_width=True)
            df, elapsed = current["df"], current["elapsed"]
            a, b, c, d = st.columns(4)
            a.metric("Objects detected", len(df))
            b.metric("Object classes", int(df["Object"].nunique()) if not df.empty else 0)
            c.metric("Avg. confidence", f"{df['Confidence'].mean():.1%}" if not df.empty else "—")
            d.metric("Analysis time", f"{elapsed:.0f} ms")
            st.subheader("Visual Insights Summary")
            st.markdown(summary_text(df))
            if not df.empty:
                st.subheader("Identified Objects Table")
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.bar_chart(df["Object"].value_counts())
                st.download_button("Download detection results (CSV)", df.to_csv(index=False).encode("utf-8"),
                                   file_name="vision_detections.csv", mime="text/csv")
            else:
                st.warning("No detections found.")
    else:
        st.info("Upload a photo to begin. Try a street scene, a room, a group photo, or an image containing everyday objects.")

elif page == "Detection History":
    st.subheader("Recent Analysis History")
    if st.session_state.history:
        hist = pd.DataFrame(st.session_state.history)
        st.dataframe(hist, use_container_width=True, hide_index=True)
        st.download_button("Export history CSV", hist.to_csv(index=False).encode("utf-8"),
                           "detection_history.csv", "text/csv")
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No analysis runs in this session yet. Analyze an image first.")

elif page == "AI Performance":
    st.subheader("AI Performance")
    st.write("Run a small benchmark on several images to compare inference time and detection volume.")
    files_uploaded = st.file_uploader("Upload multiple benchmark images", type=["jpg", "jpeg", "png", "webp"],
                                      accept_multiple_files=True)
    if st.button("Run Benchmark Evaluation", type="primary", disabled=not files_uploaded):
        records = []
        progress = st.progress(0)
        for i, f in enumerate(files_uploaded):
            try:
                image = Image.open(f).convert("RGB")
                _, df, elapsed = run_detection(image, model_name, confidence)
                records.append({"Image": f.name, "Objects": len(df),
                                "Classes": int(df["Object"].nunique()) if not df.empty else 0,
                                "Avg confidence": round(float(df["Confidence"].mean()), 3) if not df.empty else 0.0,
                                "Inference time (ms)": round(elapsed, 1)})
            except Exception as e:
                records.append({"Image": f.name, "Error": str(e)})
            progress.progress((i + 1) / len(files_uploaded))
        if records:
            benchmark = pd.DataFrame(records)
            st.dataframe(benchmark, use_container_width=True, hide_index=True)
            valid = benchmark[benchmark["Inference time (ms)"].notna()] if "Inference time (ms)" in benchmark else pd.DataFrame()
            if not valid.empty:
                x, y = st.columns(2)
                x.metric("Images evaluated", len(valid))
                y.metric("Mean inference time", f"{valid['Inference time (ms)'].mean():.1f} ms")
                st.download_button("Download benchmark CSV", benchmark.to_csv(index=False).encode("utf-8"),
                                   "vision_benchmark.csv", "text/csv")
    st.caption("Note: this is a runtime benchmark, not an accuracy benchmark. Accuracy evaluation requires labeled ground-truth annotations.")

elif page == "System Status":
    st.subheader("System Status")
    st.write("Service health for the local demo application.")
    for service, detail in [
        ("Image Analysis Service", "Pretrained YOLO object detection"),
        ("Image Processing", "Image loading and bounding-box visualization"),
        ("Insights Service", "Detection counts and confidence summaries"),
        ("Session History", "In-memory analysis history"),
    ]:
        st.success(f"**{service}** — Available\n\n{detail}")
    st.warning("The model is loaded when you run the first detection. Inference speed depends on your CPU/GPU and image size.")

elif page == "About Project":
    st.subheader("Project Overview")
    st.markdown("""
**Intelligent Vision Agent** is a computer-vision demo that combines a pretrained object-detection model with a simple analysis interface.

**Pipeline**
1. **Image input agent** — accepts and validates an uploaded image.
2. **Detection agent** — runs a pretrained YOLO model and returns bounding boxes, labels, and confidence scores.
3. **Insight agent** — summarizes detected classes and confidence.
4. **Monitoring agent** — records per-run timing and supports a multi-image runtime benchmark.

**Technology stack:** Python, Streamlit, Ultralytics YOLO, Pillow, Pandas.

**Important:** this starter project uses a pretrained object detector. The “agents” are modular pipeline stages, not independent autonomous LLM agents. It does not identify a person's identity or infer sensitive traits.
""")
    st.code("streamlit run app.py", language="bash")
