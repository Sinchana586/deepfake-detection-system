from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from src.deepshield.inference import predict_image


MODEL_PATH = Path(__file__).parent / "artifacts" / "full-shuffled" / "best.keras"

st.set_page_config(page_title="DeepShield AI", page_icon="🛡️")
st.title("DeepShield AI")
st.caption(
    "Predictions are estimates and can be wrong. "
    "They are not proof that an image is authentic or manipulated."
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

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
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
            st.caption("Prediction estimate: this result can be wrong.")
    except Exception as error:
        st.error(f"Could not analyze this image: {error}")