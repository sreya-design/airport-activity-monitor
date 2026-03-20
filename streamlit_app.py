import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["OCV_AVOID_IMSHOW"] = "1"
import streamlit as st
import numpy as np
from PIL import Image
import tempfile


st.set_page_config(
    page_title="AeroScan — Aircraft Detector",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background-color: #0a0e1a;
    color: #e8eaf0;
}

/* Hide streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1200px; }

/* Hero header */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
    position: relative;
}

.hero-badge {
    display: inline-block;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.3);
    color: #00d4ff;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    padding: 0.3rem 1rem;
    border-radius: 2px;
    margin-bottom: 1.2rem;
}

.hero-title {
    font-size: 4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1;
    margin: 0.5rem 0;
    background: linear-gradient(135deg, #ffffff 0%, #00d4ff 50%, #7c6fff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-size: 1rem;
    color: #6b7280;
    font-weight: 400;
    margin-top: 0.8rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.05em;
}

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.3), transparent);
    margin: 1.5rem 0;
}

/* Upload zone */
.upload-section {
    background: rgba(255,255,255,0.02);
    border: 1px dashed rgba(0, 212, 255, 0.25);
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    transition: border-color 0.3s;
}

/* Override streamlit file uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px dashed rgba(0, 212, 255, 0.25) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(0, 212, 255, 0.6) !important;
}

/* Stat cards */
.stats-row {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0;
}

.stat-card {
    flex: 1;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}

.stat-number {
    font-size: 2rem;
    font-weight: 800;
    color: #00d4ff;
    font-family: 'Space Mono', monospace;
    line-height: 1;
}

.stat-label {
    font-size: 0.7rem;
    color: #6b7280;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.4rem;
    font-family: 'Space Mono', monospace;
}

/* Image containers */
.img-container {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem;
    position: relative;
    overflow: hidden;
}

.img-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.img-label::before {
    content: '';
    display: inline-block;
    width: 6px;
    height: 6px;
    background: #00d4ff;
    border-radius: 50%;
}

/* Detection table */
.detection-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1.5rem;
}

.detection-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.detection-header::before {
    content: '';
    display: inline-block;
    width: 6px;
    height: 6px;
    background: #7c6fff;
    border-radius: 50%;
}

.det-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.8rem 1rem;
    border-radius: 8px;
    margin-bottom: 0.5rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    transition: background 0.2s;
}

.det-row:hover { background: rgba(0, 212, 255, 0.05); }

.det-icon {
    font-size: 1.2rem;
    width: 2rem;
    text-align: center;
}

.det-name {
    flex: 1;
    font-weight: 600;
    font-size: 0.9rem;
    color: #e8eaf0;
}

.det-conf {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #00d4ff;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.2);
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
}

.det-loc {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #4b5563;
}

/* Confidence bar */
.conf-bar-bg {
    width: 80px;
    height: 4px;
    background: rgba(255,255,255,0.08);
    border-radius: 2px;
    overflow: hidden;
}

.conf-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #7c6fff, #00d4ff);
    border-radius: 2px;
}

/* Tip box */
.tip-box {
    background: rgba(124, 111, 255, 0.07);
    border: 1px solid rgba(124, 111, 255, 0.2);
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin-top: 1.5rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #9ca3af;
    line-height: 1.7;
}

.tip-box strong {
    color: #7c6fff;
}

/* Warning / empty state */
.empty-state {
    text-align: center;
    padding: 3rem;
    color: #4b5563;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
}

/* Streamlit image caption override */
[data-testid="stImage"] { border-radius: 8px; overflow: hidden; }

/* Spinner */
[data-testid="stSpinner"] { color: #00d4ff !important; }

/* Section headings */
.section-title {
    font-size: 0.65rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 1rem;
}

/* No aircraft warning */
.no-detect {
    background: rgba(255, 100, 80, 0.06);
    border: 1px solid rgba(255, 100, 80, 0.2);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #f87171;
    text-align: center;
    margin-top: 1rem;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.2); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ COMPUTER VISION SYSTEM v1.0</div>
    <div class="hero-title">AERO SCAN</div>
    <div class="hero-sub">Aerial aircraft detection &amp; classification · YOLO26 + EfficientNet</div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    from ultralytics import YOLO
    import torch
    import torchvision.transforms as T
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    detector = YOLO("best.pt")

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    classifier = efficientnet_b0(weights=weights)
    classifier.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return detector, classifier, transform


AIRCRAFT_TYPES = [
    "Boeing 737", "Boeing 747", "Boeing 777",
    "Airbus A320", "Airbus A380", "Cessna",
    "Fighter Jet", "Cargo Plane", "Helicopter", "Unknown"
]

AIRCRAFT_ICONS = {
    "Boeing 737": "🛫", "Boeing 747": "✈️", "Boeing 777": "🛩️",
    "Airbus A320": "🛬", "Airbus A380": "✈️", "Cessna": "🛩️",
    "Fighter Jet": "🚀", "Cargo Plane": "📦", "Helicopter": "🚁", "Unknown": "❓"
}


def classify_crop(crop, classifier, transform):
    import torch
    tensor = transform(crop).unsqueeze(0)
    with torch.no_grad():
        out = classifier(tensor)
    idx = out.argmax().item() % len(AIRCRAFT_TYPES)
    return AIRCRAFT_TYPES[idx]


def run_pipeline(image, detector, classifier, transform):
    from PIL import ImageDraw

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        image.save(f.name)
        temp_path = f.name

    results = detector(temp_path, device="cpu")[0]
    os.unlink(temp_path)

    draw = ImageDraw.Draw(image)
    detections = []

    if len(results.boxes) == 0:
        return image, []

    for box, conf in zip(results.boxes.xyxy, results.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image.crop((x1, y1, x2, y2))
        label = classify_crop(crop, classifier, transform)
        confidence = float(conf)

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="#00d4ff", width=2)
        label_w = len(label) * 7 + 8
        draw.rectangle([x1, y1 - 20, x1 + label_w, y1], fill="#00d4ff")
        draw.text((x1 + 4, y1 - 17), label, fill="#0a0e1a")

        detections.append({
            "icon": AIRCRAFT_ICONS.get(label, "✈️"),
            "label": label,
            "confidence": confidence,
            "bbox": (x1, y1, x2, y2)
        })

    return image, detections


# ── Upload section ────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop an aerial or satellite image here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

st.markdown("""
<div class="tip-box">
    <strong>TIP —</strong> Best results with satellite images of airports.
    Try a Google Maps screenshot in satellite mode over JFK, Heathrow, or Mumbai Airport.
</div>
""", unsafe_allow_html=True)


# ── Main content ──────────────────────────────────────────────────────────────
if uploaded:
    image = Image.open(uploaded).convert("RGB")

    with st.spinner("Loading models..."):
        detector, classifier, transform = load_models()

    with st.spinner("Scanning for aircraft..."):
        result_img, detections = run_pipeline(
            image.copy(), detector, classifier, transform
        )

    # Stats row
    conf_avg = sum(d["confidence"] for d in detections) / len(detections) if detections else 0
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-number">{len(detections)}</div>
            <div class="stat-label">Aircraft Found</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{conf_avg:.0%}</div>
            <div class="stat-label">Avg Confidence</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{image.size[0]}×{image.size[1]}</div>
            <div class="stat-label">Image Resolution</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">YOLO26</div>
            <div class="stat-label">Model</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Images side by side
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown('<div class="img-label">INPUT IMAGE</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with col2:
        st.markdown('<div class="img-label">DETECTION OUTPUT</div>', unsafe_allow_html=True)
        st.image(result_img, use_container_width=True)

    # Detection results
    if detections:
        st.markdown('<div class="detection-card">', unsafe_allow_html=True)
        st.markdown('<div class="detection-header">DETECTION LOG</div>', unsafe_allow_html=True)

        for i, d in enumerate(detections):
            x1, y1, x2, y2 = d["bbox"]
            conf_pct = int(d["confidence"] * 100)
            st.markdown(f"""
            <div class="det-row">
                <div class="det-icon">{d["icon"]}</div>
                <div class="det-name">{d["label"]}</div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{conf_pct}%"></div>
                </div>
                <div class="det-conf">{conf_pct}%</div>
                <div class="det-loc">({x1},{y1})→({x2},{y2})</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="no-detect">
            ⚠ &nbsp; No aircraft detected — try a clearer aerial image of an airport
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="empty-state">
        ↑ upload an aerial image to begin scanning
    </div>
    """, unsafe_allow_html=True)
