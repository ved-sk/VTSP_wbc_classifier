import os
#import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------
# 1. PAGE SETUP & CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="WBC Classifier", layout="centered")

st.title("White Blood Cell Classifier")
st.write("Upload an image of a white blood cell to classify and save it.")

#MODEL_PATH = "wbc_model.h5"

#DRIVE_FILE_ID = "1bI1Q3bVkW5HpbLmRONz6wK8uDR4ukvId"
#DRIVE_LINK = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"


CLASS_NAMES = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
CONFIDENCE_THRESHOLD = 75.0  # Day 4 safety threshold percentage

# ---------------------------------------------------------
# 2. MODEL LOADING ENGINE
# ---------------------------------------------------------
@tf.keras.utils.register_keras_serializable(package="builtins", name="preprocess_input")
def preprocess_input(x):
    return x

MODEL_PATH = "wbc_model.h5"

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"⚠️ Model file '{MODEL_PATH}' was not found in the repository root directory.")
        return None
    try:
        custom_dict = {
            "preprocess_input": preprocess_input,
            "function": preprocess_input
        }
        return tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects=custom_dict)
    except Exception as e:
        st.error(f"Failed to load model file: {e}")
        return None

model = load_my_model()

if model is not None:
    st.sidebar.success("✅ Model loaded successfully!")
else:
    st.sidebar.warning("⚠️ Model not loaded. Check model path or Google Drive ID.")

# ---------------------------------------------------------
# 3. CORE INTERFACE ACTIONS (Days 2 - 5 Integration)
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Choose a WBC image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image Preview", use_container_width=True)

    if st.button("Classify and Save Image"):
        if model is None:
            st.error("Cannot perform classification because the model is not loaded.")
        else:
            with st.spinner("Analyzing image..."):
                # Save uploaded image locally
                os.makedirs("saved_images", exist_ok=True)
                save_path = os.path.join("saved_images", uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Image saved to `{save_path}`")

                # Preprocess image matrix for TensorFlow
                img_resized = image.resize((224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img_resized) / 255.0
                img_array = tf.expand_dims(img_array, 0)

                # Execute model prediction
                raw_preds = model.predict(img_array)[0]
                predicted_class_idx = int(np.argmax(raw_preds))
                confidence_score = float(np.max(raw_preds) * 100)

                if predicted_class_idx < len(CLASS_NAMES):
                    result_label = CLASS_NAMES[predicted_class_idx]
                else:
                    result_label = f"Unknown Index ({predicted_class_idx})"

                st.divider()
                st.subheader("Classification Results")

                # Day 4: Human-in-the-Loop Safety Net
                if confidence_score < CONFIDENCE_THRESHOLD:
                    st.error("⚠️ **Human Review Required**")
                    st.warning(
                        f"The model certainty ({confidence_score:.2f}%) is below the safety cutoff "
                        f"({CONFIDENCE_THRESHOLD:.0f}%). This image must be reviewed by a specialist."
                    )
                else:
                    st.metric(label="Predicted WBC Type", value=result_label)
                    st.write(f"**Model Certainty:** {confidence_score:.2f}%")

                st.caption("Note: Certainty represents mathematical confidence, not a guaranteed diagnosis.")

                # Day 3: Class Breakdown Visuals
                st.subheader("Category Breakdown")
                for idx, name in enumerate(CLASS_NAMES):
                    prob = float(raw_preds[idx])
                    st.write(f"**{name}**: {prob * 100:.2f}%")
                    st.progress(prob)
