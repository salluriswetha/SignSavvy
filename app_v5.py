
import streamlit as st
import numpy as np
import mediapipe as mp
import pickle
import sys
from pathlib import Path
from io import BytesIO
from PIL import Image

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="SignSavvy",
    page_icon="🤟",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sign_model.pkl"
TASK_PATH = BASE_DIR / "hand_landmarker.task"

st.title("🤟 SignSavvy")
st.write("Capture your hand sign to get a prediction.")

# --------------------------------------------------
# Check required files
# --------------------------------------------------
if not MODEL_PATH.exists():
    st.error("sign_model.pkl file not found.")
    st.stop()

if not TASK_PATH.exists():
    st.error("hand_landmarker.task file not found.")
    st.stop()

# --------------------------------------------------
# Load the trained model
# Compatibility alias for some NumPy 2.x pickles
# --------------------------------------------------
@st.cache_resource
def load_model(model_path):
    import numpy as np

    # Support pickles referencing NumPy 2.x module paths
    try:
        import numpy._core.numeric
    except ImportError:
        sys.modules.setdefault("numpy._core", np.core)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)

    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)

try:
    model = load_model(str(MODEL_PATH))
except Exception as error:
    st.error("Could not load sign_model.pkl.")
    st.code(str(error))
    st.info(
        "The model may need to be re-saved using a compatible "
        "NumPy and scikit-learn environment."
    )
    st.stop()

# --------------------------------------------------
# Create MediaPipe hand landmarker
# --------------------------------------------------
@st.cache_resource
def create_landmarker(task_path):
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=task_path
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1
    )

    return vision.HandLandmarker.create_from_options(options)

try:
    landmarker = create_landmarker(str(TASK_PATH))
except Exception as error:
    st.error("Could not initialize the hand detector.")
    st.code(str(error))
    st.stop()

# --------------------------------------------------
# Capture an image
# --------------------------------------------------
image = st.camera_input("Capture your hand sign")

if image is not None:
    try:
        rgb_image = np.array(
            Image.open(BytesIO(image.getvalue())).convert("RGB")
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        detection_result = landmarker.detect(mp_image)

        if not detection_result.hand_landmarks:
            st.warning(
                "No hand detected. Keep your hand clearly visible "
                "and try again."
            )
        else:
            # 21 hand landmarks, each with x, y, z coordinates
            landmarks = detection_result.hand_landmarks[0]

            features = []
            for landmark in landmarks:
                features.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            # The model must have been trained using the same
            # 63-feature format and ordering.
            expected_features = getattr(
                model, "n_features_in_", len(features)
            )

            if len(features) != expected_features:
                st.error(
                    f"Feature mismatch: extracted {len(features)} "
                    f"values, but the model expects "
                    f"{expected_features}."
                )
            else:
                prediction = model.predict([features])[0]

                st.success(f"Predicted sign: {prediction}")

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba([features])[0]
                    confidence = float(np.max(probabilities)) * 100

                    st.metric(
                        "Prediction confidence",
                        f"{confidence:.2f}%"
                    )

                    st.caption(
                        "Confidence is the model's estimated "
                        "probability, not a guarantee of correctness."
                    )

    except Exception as error:
        st.error("Something went wrong while processing the image.")
        st.code(str(error))
