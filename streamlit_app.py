import streamlit as st
from PIL import Image
import os

st.title("White Blood Cell Classifier")
st.write("Upload an image of a white blood cell to classify and save it.")

# 1. File Uploader Interface
uploaded_file = st.file_uploader("Choose a WBC image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Create a save button
    if st.button("Save Image"):
        # Create a directory to save images if it doesn't exist
        os.makedirs("saved_images", exist_ok=True)
        
        # Save the file locally on the server container
        save_path = os.path.join("saved_images", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.success(f"Successfully saved to {save_path}!")
        
        # TODO: Insert your model classification function here:
        # prediction = your_model.predict(image)
        # st.write(f"Result: {prediction}")

