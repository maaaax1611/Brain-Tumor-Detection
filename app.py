import time
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image
import imutils
import cv2
import io

st.title("🧠 Brain Tumor Classifier")
st.markdown("Upload an **MRI brain scan**, and let the model predict whether a **tumor** is present.")

col1, col2 = st.columns([1, 2])

# file upload
uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

# load model
@st.cache_resource
def load_model(model_path):
    base_model = VGG19(include_top=False, input_shape=(240,240,3))
    base_model_layer_names = [layer.name for layer in base_model.layers]

    x=base_model.output
    flat = Flatten()(x)

    class_1 = Dense(4608, activation = 'relu')(flat)
    drop_out = Dropout(0.2)(class_1)
    class_2 = Dense(1152, activation = 'relu')(drop_out)
    output = Dense(2, activation = 'softmax')(class_2)

    model = Model(base_model.inputs, output)
    model.load_weights(model_path)
    return model

model_path = "model/vgg19_unfrozen_80e_b32.h5"
model = load_model(model_path)

# preprocessing
def preprocess_image(image, plot=False):
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    thres = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thres =cv2.erode(thres, None, iterations = 2)
    thres = cv2.dilate(thres, None, iterations = 2)

    cnts = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key = cv2.contourArea)

    ext_left = tuple(c[c[:,:,0].argmin()][0])
    ext_right = tuple(c[c[:,:,0].argmax()][0])
    ext_top = tuple(c[c[:,:,1].argmin()][0])
    ext_bot = tuple(c[c[:,:,1].argmax()][0])

    new_image = image[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]]
     # scale image to 240x240
    new_image = cv2.resize(new_image, (240, 240))
    # scale pixel values [0,1]
    new_image = new_image.astype('float32') / 255.0
    # Add batch dimension: (1, 240, 240, 3)
    new_image = np.expand_dims(new_image, axis=0)

    return new_image


# make prediction
if uploaded_file is not None:

    with col1:
        st.image(Image.open(uploaded_file), caption="Uploaded Image", use_container_width=True)

    # preprocess the image
    processed_image = preprocess_image(uploaded_file)

    with col2:
        with st.spinner("Analyzing image..."):
            time.sleep(2)
            prediction = model.predict(processed_image)[0][0]

        st.markdown("Prediction:")
        if prediction < 0.5:
            st.error("Tumor detected (Confidence: {:.2f}%)".format((1-prediction) * 100))
        else:
            st.success("No tumor detected (Confidence: {:.2f}%)".format((prediction) * 100))

        confidence = (1 - prediction) if prediction < 0.5 else prediction
        st.progress(confidence)