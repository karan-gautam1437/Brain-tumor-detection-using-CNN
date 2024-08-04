import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load the trained model
model = load_model('model_weights/vgg19_model_02.h5')

# Function to preprocess image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(240, 240))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Streamlit app
st.title('Brain Tumor Detection')
st.write('Upload an MRI image to detect if it has a brain tumor.')

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg")

if uploaded_file is not None:
    # Display uploaded image
    image = load_img(uploaded_file, target_size=(240, 240))
    st.image(image, caption='Uploaded MRI image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the uploaded image
    img_array = preprocess_image(uploaded_file)

    # Make predictions
    prediction = model.predict(img_array)
    class_names = ['Non-Tumorous', 'Tumorous']
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f'Prediction: {predicted_class}')

    # Display prediction probability
    st.write(f'Prediction Confidence: {prediction[0][np.argmax(prediction)] * 100:.2f}%')
