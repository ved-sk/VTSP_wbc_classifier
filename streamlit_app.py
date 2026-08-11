import streamlit as st
from PIL import Image
import os
import tensorflow as tf
import numpy as np
import gdown

# 1. Main Titles
st.title("White Blood Cell Classifier")
st.write("Upload an image of a white blood cell to classify and save it.")

# 2. File and Link Configurations
MODEL_PATH = "wbc_model.keras"
DRIVE_LINK = "https://google.com"

# 3. Model Loading Function
@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading trained model weights from Google Drive... Please wait."):
            try:
                gdown.download(DRIVE_LINK, MODEL_PATH, quiet=False)
            except Exception as e:
                st.error(f"Download failed: {e}")
                return None
    return tf.keras.models.load_model(MODEL_PATH)

# Initialize model globally
# model = load_my_model()

# 4. Image Uploader UI Element
uploaded_file = st.file_uploader("Choose a WBC image...", type=["jpg", "jpeg", "png"])

# 5. Core Interface Actions
if uploaded_file is not None and model is not None:
    # Read and display the image immediately upon upload
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Check if the user clicks the action button
    if st.button("Classify and Save Image"):
        
        # Save image inside the server's workspace folder
        os.makedirs("saved_images", exist_ok=True)
        save_path = os.path.join("saved_images", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Successfully saved to {save_path}!")
        
        # Preprocess image matrix for TensorFlow
        img_resized = image.resize((224, 224)) 
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0) # Format as an image batch
        
predictions = model.predict(img_array)
        
        # Display raw output values
st.write(f"Raw Prediction Outputs: {predictions}")
