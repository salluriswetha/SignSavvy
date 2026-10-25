
import base64
import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
import pickle
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.set_page_config(
    page_title="SignSavvy",
    page_icon="signsavvy_logo.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {
    max-width: 1450px;
    padding-top: 1rem;
    padding-bottom: 0.2rem;
}
.stApp {
    background: #07111f;
}
.subtitle {
    text-align: center;
    color: #9caec5;
    font-size: 15px;
    margin-top: 8px;
    margin-bottom: 22px;
}
.capture-title {
    color: white;
    font-size: 27px;
    font-weight: 700;
    margin-bottom: 8px;
}
.prediction-title {
    color: white;
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 10px;
}
.prediction-card {
    background: linear-gradient(145deg,#0e2034,#0b1829);
    border: 1px solid #294a69;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    margin-bottom: 15px;
}
.prediction-label {
    color: #8fa6c1;
    font-size: 12px;
    margin-bottom: 5px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.prediction-value {
    color: white;
    font-size: 56px;
    font-weight: 800;
    line-height: 1.05;
}
.confidence {
    color: #36a9ff;
    font-size: 16px;
    font-weight: 600;
    margin-top: 7px;
}
.confidence-track {
    width: 100%;
    height: 7px;
    background: #182c42;
    border-radius: 10px;
    margin-top: 12px;
    overflow: hidden;
}
.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg,#237fc4,#36a9ff);
    border-radius: 10px;
}
.history-title {
    color: white;
    font-size: 17px;
    font-weight: 700;
    margin-top: 6px;
    margin-bottom: 8px;
}
.history-item {
    background: #0d1b2d;
    border: 1px solid #1e3654;
    border-radius: 9px;
    padding: 9px 12px;
    margin-bottom: 6px;
    color: #dbe7f5;
    font-size: 13px;
}
.history-sign {
    color: white;
    font-weight: 700;
    font-size: 15px;
}
.history-confidence {
    color: #8fa6c1;
    float: right;
}
.stButton > button {
    width: 100%;
    border-radius: 9px;
    border: 1px solid #31577e;
    background: #10243b;
    color: white;
    font-weight: 600;
    padding: .4rem .7rem;
}
.stButton > button:hover {
    border-color: #36a9ff;
    color: #36a9ff;
    background: #112b45;
}
[data-testid="stCameraInput"] {
    border-radius: 13px;
    overflow: hidden;
}
[data-testid="stCameraInput"] img {
    max-height: 390px !important;
    width: 100% !important;
    object-fit: cover !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- Centered Logo ----------
with open("signsavvy_logo.png", "rb") as logo_file:
    logo_data = base64.b64encode(logo_file.read()).decode("utf-8")

st.html(f"""
<div style="
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
    margin: 0 auto;
">
    <img
        src="data:image/png;base64,{logo_data}"
        style="
            width: 220px;
            height: auto;
            display: block;
            margin: 0 auto;
        "
    />
</div>
""")

st.markdown(
    '<div class="subtitle">Real-Time Sign Language Recognition Using AI</div>',
    unsafe_allow_html=True
)

# ---------- Session State ----------
if "history" not in st.session_state:
    st.session_state["history"] = []

if "last_capture_id" not in st.session_state:
    st.session_state["last_capture_id"] = None

if st.session_state.pop("clear_history_now", False):
    st.session_state["history"] = []
    st.session_state["last_capture_id"] = None

# ---------- Load Model ----------
with open("sign_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# ---------- MediaPipe ----------
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)


def extract_features(landmarks):
    wrist = landmarks[0]
    features = []

    for landmark in landmarks:
        features.extend([
            landmark.x - wrist.x,
            landmark.y - wrist.y,
            landmark.z - wrist.z
        ])

    values = np.array(features, dtype=np.float32)
    scale = np.max(np.abs(values))

    if scale > 0:
        values = values / scale

    return values


# ---------- Main Layout ----------
left_col, right_col = st.columns([1.65, 0.85], gap="large")

with left_col:
    st.markdown(
        '<div class="capture-title">Sign Capture</div>',
        unsafe_allow_html=True
    )

    image = st.camera_input(
        "Capture your hand sign",
        key="sign_camera",
        label_visibility="collapsed"
    )

    if image is not None:
        capture_id = hash(image.getvalue())

        if capture_id != st.session_state["last_capture_id"]:
            st.session_state["last_capture_id"] = capture_id

            file_bytes = np.asarray(
                bytearray(image.getvalue()),
                dtype=np.uint8
            )
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if frame is not None:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_image
                )
                result = detector.detect(mp_image)

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    features = extract_features(landmarks).reshape(1, -1)

                    prediction = model.predict(features)[0]
                    probabilities = model.predict_proba(features)[0]
                    confidence = float(np.max(probabilities)) * 100

                    st.session_state["history"].append(
                        (str(prediction), round(confidence, 1))
                    )
                else:
                    st.warning(
                        "No hand detected. Position your hand clearly "
                        "and capture the sign again."
                    )

with right_col:
    st.markdown(
        '<div class="prediction-title">AI Prediction</div>',
        unsafe_allow_html=True
    )

    history = st.session_state["history"]

    if history:
        latest_sign, latest_confidence = history[-1]
        confidence_width = min(max(latest_confidence, 0), 100)

        st.html(f"""
        <div class="prediction-card">
            <div class="prediction-label">Detected Sign</div>
            <div class="prediction-value">{latest_sign}</div>
            <div class="confidence">
                Confidence: {latest_confidence:.1f}%
            </div>
            <div class="confidence-track">
                <div class="confidence-fill"
                     style="width:{confidence_width}%;"></div>
            </div>
        </div>
        """)
    else:
        st.html("""
        <div class="prediction-card">
            <div class="prediction-label">Detected Sign</div>
            <div class="prediction-value">—</div>
            <div class="confidence">Waiting for sign</div>
        </div>
        """)

    st.markdown(
        '<div class="history-title">Prediction History</div>',
        unsafe_allow_html=True
    )

    if history:
        for sign, confidence in reversed(history[-6:]):
            st.html(f"""
            <div class="history-item">
                <span class="history-sign">{sign}</span>
                <span class="history-confidence">{confidence:.1f}%</span>
            </div>
            """)
    else:
        st.html('<div class="history-item">No predictions yet</div>')

    if st.button("Clear History", use_container_width=True):
        st.session_state["clear_history_now"] = True
        st.rerun()

