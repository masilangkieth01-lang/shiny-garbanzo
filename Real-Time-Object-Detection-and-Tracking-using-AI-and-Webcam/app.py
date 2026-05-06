import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import av
import cv2
import threading

# --- 1. NEON MINT & DARK GRAY THEME ---
st.set_page_config(page_title="AEGIS-VISION // HUB", layout="wide")

st.markdown("""
    <style>
    /* Dark Gray Background */
    .stApp { 
        background-color: #121212; 
    }
    
    /* Neon Mint Green Headers & Text */
    h1, h2, h3, p, .stMarkdown { 
        color: #39FF14 !important; 
        font-family: 'Courier New', Courier, monospace !important;
        text-shadow: 0 0 5px rgba(57, 255, 20, 0.4);
    }
    
    /* Glassmorphism Panel Settings Background */
    [data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
        background-color: #1e1e1e;
        border: 1px solid #39FF14;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 0 15px rgba(57, 255, 20, 0.1);
    }
    
    /* Slider & Button Styling */
    .stWidget label p {
        color: #39FF14 !important;
        font-weight: bold !important;
    }
    .stButton>button { 
        border: 1px solid #39FF14; 
        color: #39FF14; 
        background-color: #121212; 
        width: 100%;
        font-weight: bold;
    }
    .stButton>button:hover { 
        background-color: #39FF14; 
        color: #121212; 
    }
    </style>
""", unsafe_allow_html=True)

# --- CACHE MODEL ---
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# --- THREAD-SAFE CACHE (For Snapshots & Sliders) ---
# Because WebRTC runs in the background, we need a secure way to 
# pass slider values in and pull image frames out.
class VideoContext:
    lock = threading.Lock()
    current_frame = None
    conf_threshold = 0.5
    iou_threshold = 0.45

# --- VIDEO PROCESSING ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Run YOLOv8 tracking using the live slider parameters
    results = model.track(
        img,
        persist=True,
        conf=VideoContext.conf_threshold,
        iou=VideoContext.iou_threshold,
        verbose=False
    )

    # Annotate frame
    annotated_frame = results[0].plot()

    # Save the latest frame safely into our VideoContext for snapshots
    with VideoContext.lock:
        VideoContext.current_frame = annotated_frame

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# --- LAYOUT ---
st.title("🟩 AEGIS-VISION // LIVE TRACING")
st.write("SYSTEM ONLINE. Point your camera at objects to identify them in real-time.")

# Split layout: Main feed (3 parts width), Settings (1 part width)
col_main, col_settings = st.columns([3, 1], gap="medium")

# --- RIGHT SIDE: PANEL SETTINGS ---
with col_settings:
    st.write("### 🎛️ SYSTEM CONTROLS")
    
    # Sliders update the global VideoContext variables instantly
    conf = st.slider("ACCURACY SENSITIVITY", 0.1, 1.0, 0.50, 0.05)
    iou = st.slider("OVERLAP LIMIT (IoU)", 0.1, 1.0, 0.45, 0.05)
    
    VideoContext.conf_threshold = conf
    VideoContext.iou_threshold = iou
    
    st.divider()
    
    # Snapshot Documentation Tool
    st.write("### 📸 DOCUMENTATION")
    st.write("Capture the live feed to save a snapshot.")
    
    if st.button("TAKE SNAPSHOT"):
        with VideoContext.lock:
            if VideoContext.current_frame is not None:
                # Convert from OpenCV's BGR format to Streamlit's RGB format
                rgb_img = cv2.cvtColor(VideoContext.current_frame, cv2.COLOR_BGR2RGB)
                st.image(rgb_img, caption="TARGET LOGGED", use_container_width=True)
            else:
                st.warning("Camera offline. Cannot capture snapshot.")

# --- LEFT SIDE: LIVE VIDEO FEED ---
with col_main:
    webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )