from pathlib import Path
from src.deepshield.inference import predict_image, predict_video
import tempfile
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from src.deepshield.inference import predict_image, predict_video



MODEL_PATH = Path(__file__).parent / "artifacts" / "fast" / "best.keras"

st.set_page_config(page_title="DeepShield AI", page_icon="🛡️")
st.title("DeepShield AI")
st.caption(
    "Image screening baseline. Scores are estimates and are not proof "
    "that an image is authentic or manipulated."
)


@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)


if not MODEL_PATH.is_file():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

media_type = st.segmented_control(
    "Media type",
    options=["Image", "Video"],
    default="Image",
    key="media_type",
)

allowed_types = (
    ["jpg", "jpeg", "png", "webp"]
    if media_type == "Image"
    else ["mp4", "mov", "avi", "mkv", "webm"]
)

uploaded_file = st.file_uploader(
    f"Choose a {media_type.lower()}",
    type=allowed_types,
)

if uploaded_file is not None and media_type == "Image":
    try:
        image = Image.open(uploaded_file).convert("RGB")
        rgb_image = np.asarray(image)

        st.image(
            rgb_image,
            caption="Uploaded image",
            width="stretch",
            alt="User-uploaded image for analysis",
        )

        with st.form("image_analysis"):
            analyze_image = st.form_submit_button("Analyze image", type="primary")

        if analyze_image:
            with st.spinner("Analyzing image…"):
                fake_probability = predict_image(model, rgb_image)

            if fake_probability >= 0.5:
                prediction = "FAKE"
                probability = fake_probability
            else:
                prediction = "REAL"
                probability = 1.0 - fake_probability

            st.markdown(f"### Prediction: **{prediction}** — {probability:.1%}")
            st.caption("This preliminary model can be wrong.")
    except Exception as error:
        st.error(f"Could not analyze this image: {error}")

elif uploaded_file is not None and media_type == "Video":
    temp_path = None

    try:
        suffix = Path(uploaded_file.name).suffix or ".mp4"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        if st.button("Analyze video", type="primary"):
            with st.spinner("Sampling and analyzing video frames…"):
                result = predict_video(model, temp_path, max_frames=24)

            st.metric(
                "Mean sampled-frame fake score",
                f"{result['mean_fake_probability']:.1%}",
            )
            st.dataframe(result["frames"], hide_index=True, width="stretch")
            st.caption(
                "The mean is a simple summary of sampled frames, "
                "not a calibrated probability for the whole video."
            )

    except Exception as error:
        st.error(f"Could not analyze this video: {error}")

    finally:
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink()