import streamlit as st
from PIL import Image
import os

st.title("White Blood Cell Classifier")
st.write("Upload an image of a white blood cell to classify and save it.")

MODEL_PATH = "wbc_model.keras"
# Replace the string below with your exact copied Google Drive share link
DRIVE_LINK = "https://drive.google.com/file/d/1gDXUkR-nZtbrJSjnUfSYG51X4Ug7Yljy/view?usp=drive_link"


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
        # --- PREDICTION LAYER ---
        # 1. Predict using your loaded Keras model
        predictions = model.predict(img_array)
        
        # 2. Extract the highest probability score and its index
        predicted_class_idx = np.argmax(predictions[0])
        confidence_score = np.max(predictions[0]) * 100  # Convert decimal to percentage
        
        # 3. Define your exact dataset categories (REPLACE THESE WITH YOUR EXACT MODEL CLASSES!)
        CLASS_NAMES = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
        
        # 4. Safely pull the corresponding category text string
        if predicted_class_idx < len(CLASS_NAMES):
            result_label = CLASS_NAMES[predicted_class_idx]
        else:
            result_label = f"Unknown Class Index ({predicted_class_idx})"
            
        # --- BEAUTIFUL DISPLAY INTERFACE ---
        st.write("---")
        st.subheader("Classification Results")
        
        # Highlight the prediction with a bold callout box
        st.metric(label="Predicted WBC Type", value=result_label)
        
        # Show the accuracy metric as a progress bar or text badge
        st.write(f"**Confidence Score / Model Certainty:** {confidence_score:.2f}%")
        st.progress(int(confidence_score) / 100)
        
        # Optional debug layer: expand to see all category raw outputs
        with st.expander("See Raw Probability Breakdown"):
            for name, score in zip(CLASS_NAMES, predictions[0]):
                st.write(f"**{name}**: {score * 100:.2f}%")

