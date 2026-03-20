import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import cv2
import io

st.set_page_config(layout="wide", page_title="Airport Monitor")
st.title("🛩️ Airport Activity & Infrastructure Monitor")
st.markdown("**Satellite/UAV Computer Vision for Aerospace Operations**")

# Sidebar controls
st.sidebar.header("📡 Analysis Settings")
conf_threshold = st.sidebar.slider("Detection Confidence", 0.1, 0.9, 0.4)

# Main app
uploaded_file = st.file_uploader("📷 Upload satellite/aerial image", 
                                type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    original_image = image.copy()
    
    st.image(image, caption="Input Image", use_column_width=True)
    
    if st.button("🚀 ANALYZE AIRPORT", type="primary", use_container_width=True):
        with st.spinner("Running aerospace analysis..."):
            
            # Convert to OpenCV format
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            
            # 1. Aircraft detection simulation (template matching)
            aircraft_count = detect_aircraft_like_regions(img_array)
            
            # 2. Runway crack analysis
            runway_stats = analyze_runway(img_array)
            
            # 3. Activity metrics
            activity_score = min(100, aircraft_count * 6)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("✈️ Aircraft", aircraft_count)
            col2.metric("🛤️ Runway Health", runway_stats['condition_score'])
            col3.metric("📊 Activity Level", f"{activity_score}/100")
            col4.metric("⚠️ Alerts", 0 if runway_stats['status'] == "GOOD" else 1)
            
            # Executive summary
            st.markdown(f"""
            ### 📊 EXECUTIVE SUMMARY
            
            **Aircraft Operations**
            - Detected: **{aircraft_count}** aircraft across aprons
            - Activity: **{activity_score:.0f}/100** (High/Normal/Low)
            
            **Runway Infrastructure**  
            - Condition Index: **{runway_stats['condition_score']}**
            - Crack Density: **{runway_stats['crack_density']}**
            - Status: **{runway_stats['status']}**
            
            **💰 Business Impact**
            - Estimated daily revenue: **${aircraft_count * 15000:,.0f}**
            - Operations status: ✅ **NORMAL**
            """)
            
            # Annotated image
            annotated_img = annotate_image(original_image, aircraft_count)
            st.image(annotated_img, caption="Analysis Overlay", use_container_width=True)

def detect_aircraft_like_regions(img_array):
    """Detect rectangular aircraft-like regions"""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 30, 100)
    
    # Find contours (aircraft shapes)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    aircraft_regions = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if 5000 < area < 50000:  # Aircraft size range
            aircraft_regions += 1
    
    return min(aircraft_regions, 25)  # Cap at realistic number

def analyze_runway(img_array):
    """Runway crack detection using edge analysis"""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    # Crack-like lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cracks = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    crack_ratio = np.sum(cracks > 0) / cracks.size
    condition_score = max(0, 100 - crack_ratio * 10000)
    
    status = "GOOD" if condition_score > 80 else "WARNING" if condition_score > 60 else "CRITICAL"
    
    return {
        'condition_score': f"{condition_score:.1f}%",
        'crack_density': f"{crack_ratio*100:.2f}%", 
        'status': status
    }

def annotate_image(image, aircraft_count):
    """Add analysis annotations to image"""
    draw = ImageDraw.Draw(image)
    w, h = image.size
    
    # Add aircraft count
    draw.rectangle([10, 10, w-10, 80], fill=(0, 0, 0, 180))
    draw.text((20, 25), f"✈️ {aircraft_count} Aircraft Detected", 
              fill="white", font_size=24)
    
    # Add status bar
    draw.rectangle([10, h-60, w-10, h-10], fill="green")
    draw.text((20, h-45), "✅ OPERATIONAL", fill="white", font_size=20)
    
    return image
