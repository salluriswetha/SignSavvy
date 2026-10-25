

import base64
import sys
import pickle
from pathlib import Path

import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sign_model.pkl"
TASK_PATH = BASE_DIR / "hand_landmarker.task"
LOGO_PATH = BASE_DIR / "signsavvy_logo.png"

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="SignSavvy",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🤟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# STYLING
# --------------------------------------------------
st.markdown("""
<style>
.block-container {
    max-width: 1450px;
    padding-top: 1rem;
    padding-bottom: 0.2rem;
}
.stApp { background: #07111f; }
.subtitle {
    text-align: center; color: #9caec5; font-size: 15px;
    margin-top: 8px; margin-bottom: 22px;
}
.capture-title {
    color: white; font-size: 27px; font-weight: 700;
    margin-bottom: 8px;
}
.prediction-title {
    color: white; font-size: 21px; font-weight: 700;
    margin-bottom: 10px;
}
.prediction-card {
    background: linear-gradient(145deg,#0e2034,#0b1829);
    border: 1px solid #294a69; border-radius: 16px;
    padding: 20px; text-align: center; margin-bottom: 15px;
}
.prediction-label {
    color: #8fa6c1; font-size: 12px; margin-bottom: 5px;
    text-transform: uppercase; letter-spacing: 1px;
}
.prediction-value {
    color: white; font-size: 56px; font-weight: 800;
    line-height: 1.05;
}
.confidence {
    color: #36a9ff; font-size: 16px; font-weight: 600;
    margin-top: 7px;
}
.confidence-track {
    width: 100%; height: 7px; background: #182c42;
    border-radius: 10px; margin-top: 12px; overflow: hidden;
}
.confidence-fill {
    height: 100%; background: linear-gradient(90deg,#237fc4,#36a9ff);
    border-radius: 10px;
}
.history-title {
    color: white; font-size: 17px; font-weight: 700;
    margin-top: 6px; margin-bottom: 8px;
}
.history-item {
    background: #0d1b2d; border: 1px solid #1e3654;
    border-radius: 9px; padding: 9px 12px; margin-bottom: 6px;
    color: #dbe7f5; font-size: 13px;
}
.history-sign { color: white; font-weight: 700; font-size: 15px; }
.history-confidence { color: #8fa6c1; float: right; }
.stButton > button {
    width: 100%; border-radius: 9px; border: 1px solid #31577e;
    background: #10243b; color: white; font-weight: 600;
    padding: .4rem .7rem;
}
.stButton > button:hover {
    border-color: #36a9ff; color: #36a9ff; background: #112b45;
}
[data-testid="stCameraInput"] { border-radius: 13px; overflow: hidden; }
[data-testid="stCameraInput"] img {
    max-height: 390px !important; width: 100% !important;
    object-fit: cover !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# CENTERED LOGO
# --------------------------------------------------
if LOGO_PATH.exists():
    logo_data = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
    st.html(f"""
    <div style="width:100%;display:flex;justify-content:center;
                align-items:center;text-align:center;margin:0 auto;">
      <img src="data:image/png;base64,{logo_data}"
           style="width:220px;height:auto;display:block;margin:0 auto;"/>
    </div>
    """)
else:
    st.warning("Logo file not found: signsavvy_logo.png")

st.markdown(
    '<div class="subtitle">Real-Time Sign Language Recognition Using AI</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

if "last_capture_id" not in st.session_state:
    st.session_state["last_capture_id"] = None

if st.session_state.pop("clear_history_now", False):
    st.session_state["history"] = []
    st.session_state["last_capture_id"] = None

# --------------------------------------------------
# LOAD MODEL
# NumPy 2.x pickle compatibility alias for NumPy 1.x
# --------------------------------------------------
if not MODEL_PATH.exists():
    st.error("sign_model.pkl file not found.")
    st.stop()

try:
    import numpy as np

    # Pickles created with NumPy 2.x may reference
    # numpy._core.numeric, while NumPy 1.x uses numpy.core.
    try:
        import numpy._core.numeric
    except ImportError:
        import numpy.core
        import numpy.core.numeric

        sys.modules["numpy._core"] = numpy.core
        sys.modules["numpy._core.numeric"] = numpy.core.numeric

    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)

except Exception as error:
    st.error("Could not load the trained AI model.")
    st.code(f"{type(error).__name__}: {error}")
    st.info(
        "Check NumPy and scikit-learn compatibility with the "
        "environment used to create sign_model.pkl."
    )
    st.stop()

# --------------------------------------------------
# MEDIAPIPE
# --------------------------------------------------
if not TASK_PATH.exists():
    st.error("hand_landmarker.task file not found.")
    st.stop()

@st.cache_resource
def create_detector(task_path):
    base_options = python.BaseOptions(model_asset_path=task_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1
    )
    return vision.HandLandmarker.create_from_options(options)

try:
    detector = create_detector(str(TASK_PATH))
except Exception as error:
    st.error("Could not initialize MediaPipe hand detection.")
    st.code(f"{type(error).__name__}: {error}")
    st.stop()

# --------------------------------------------------
# FEATURE EXTRACTION
# Must match training code exactly
# --------------------------------------------------
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

# --------------------------------------------------
# MAIN LAYOUT
# --------------------------------------------------
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

            try:
                file_bytes = np.frombuffer(
                    image.getvalue(), dtype=np.uint8
                )
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if frame is None:
                    st.error("Could not read the captured image.")
                else:
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_image
                    )
                    result = detector.detect(mp_image)

                    if result.hand_landmarks:
                        landmarks = result.hand_landmarks[0]
                        features = extract_features(landmarks).reshape(1, -1)

                        expected = getattr(model, "n_features_in_", 63)
                        if features.shape[1] != expected:
                            st.error(
                                f"Feature mismatch: model expects {expected}, "
                                f"but extracted {features.shape[1]}."
                            )
                        else:
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

            except Exception as error:
                st.error("Image prediction failed.")
                st.code(f"{type(error).__name__}: {error}")

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
          <div class="confidence">Confidence: {latest_confidence:.1f}%</div>
          <div class="confidence-track">
            <div class="confidence-fill" style="width:{confidence_width}%;"></div>
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
