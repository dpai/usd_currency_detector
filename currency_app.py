from PIL import Image
import numpy as np 
import streamlit as st 
from tensorflow.keras.models import Model, load_model
import os
import cv2

# Helpers
def checkPathExists(path):
  if not os.path.exists(path):
    print(f"Cannot access path: {path}")
    return False
  else:
    print (f"Path {path} accessible")
    return True

# Function to Read and Manupilate Images
def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    img_load = True
    return image, img_load

@st.cache_resource
def load_currency_model():
    if checkPathExists('full_model.h5'):
        inference_model = load_model('full_model.h5')
        return inference_model
    else:
        st.write("No Model found. There will be no prediction")
        return None

st.title("Currency Detection - US Dollars")
with st.spinner("Load Currency Model ...."):
    inference_model = load_currency_model()
    IMG_SIZE = (224, 224)
    labels = ['100_1', '100_2', '10_1', '10_2', '1_1', '1_2', '20_1', '20_2', '2_1',
       '2_2', '50_1', '50_2', '5_1', '5_2']

# Uploading the File to the Page
col1, col2 = st.columns(2)
with col1:
    uploadFile = st.file_uploader(label="Upload a US bank note", type=['jpg', 'png'])

# Checking the Format of the page
if uploadFile is not None:
    # Perform your Manupilations (In my Case applying Filters)
    img, img_load = load_image(uploadFile)
    st.write("Image Uploaded Successfully")
    col3,col4 = st.columns(2)
    with col3:
        st.subheader('Original Image')
        st.image(img)
        st.write("Image dimensions:",img.shape)
        st.write("Max pixel values:",np.amax(img))
        if img.shape[0] != img.shape[1]:
            st.markdown("**Ensure original input is a square image!!")
    with col4:
        st.subheader('Resized Image')
        img_resize = cv2.resize(np.asarray(img), (IMG_SIZE[0], IMG_SIZE[1]))
        st.image(img_resize)
        st.write("Image dimensions:",img_resize.shape)
        st.write("Max pixel value",np.amax(img_resize))
    # Infer on the chosen Image
    if img_load and (inference_model != None):
        with st.spinner("Classifying.. "):
            img_input = np.array([img_resize/255.0])
            y_pred = inference_model.predict(img_input)
            predictions = np.argmax(y_pred, axis=1)
            predicted_denominations = labels[int(predictions)]
            st.subheader("Image Classified as")
            formatted_output = f"{predicted_denominations.split('_')[0]} Dollars"
            st.metric(label="Prection",value=formatted_output, label_visibility='hidden')

else:
    st.write("Make sure you image is in JPG/PNG Format.")