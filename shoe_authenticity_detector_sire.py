from datetime import datetime
from keras.models import load_model
import streamlit as st
import numpy as np
import glob
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps
import keras
from datetime import date
from st_btn_select import st_btn_select

selection = st_btn_select(('CHECK YOUR SHOES', 'ABOUT'))

st.markdown(
    """
<style>

    body {
  background: #ff0099; 
  background: -webkit-linear-gradient(to right, #ff0099, #493240); 
  background: linear-gradient(to right, #ff0099, #493240); 
    }
    
}



</style>
""",
    unsafe_allow_html=True,
)





if selection == 'CHECK YOUR SHOES':
    
    st.markdown(""" <style> .font {
    font-size:50px ; font-weight: 800; color: #7792E3;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Shoethentic</p>', unsafe_allow_html=True)
    
    st.header("Created by Julia Huang & Justin Huang")
    st.header("Detect if your shoes are fake or not via AI!")
    st.subheader("Quick and easy; you only need to upload images to receive an automatic result!")



    image = st.file_uploader(label = "Upload an image for analysis:", type = ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'raw', 'webp'])

    def import_and_predict(image_data, model):
        size = (227, 227)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        img = tf.keras.utils.img_to_array(image)
        img = tf.expand_dims(img, 0)
        probs = model.predict(img)
        score = tf.nn.softmax(probs[0])
        text = ("Shoethentic predicts that this is an image of **{} shoe with {:.2f}% confidence**."
        .format(class_names[np.argmax(score)], 100 * np.max(score)))
        return text

    loaded_model = tf.keras.models.load_model('model.h5')
    class_names = ['Fake', 'Real']

    predictionText = "Prediction: Waiting for an image upload"

    if image is not None:
        st.image(image)
        predictionText = (import_and_predict(Image.open(image), loaded_model))

    st.markdown(predictionText)    

    

if selection == 'ABOUT':
    st.markdown(""" <style> .font {
    font-size:50px ; font-weight: 800; color: #7792E3;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">About Shoethentic</p>', unsafe_allow_html=True)
   
    st.subheader("About the Creator")
    st.markdown("Shoethentic's web app and model is built by Julia Huang, a current student and developer at Sire, and the dataset is created by Justin Huang.")

    st.subheader("Mission")
    st.markdown("Due to the high prevalence of counterfeit shoe production, the goal of **Shoethentic** is to provide the sneakerhead community an opportunity to check the authenticity of each and every shoe they buy. **Shoethentic** aims to make this checking process simpler and more convenient by utilizing AI & machine learning.")
    st.subheader("How Shoethentic was Built")
  
    st.markdown("Shoethentic has two parts: the AI model and web app. The AI model is built using the TensorFlow framework while the web app is built using Streamlit. We trained the model on a dataset consisting of fake and real shoe images sourced from the CheckCheck mobile app.")
 
    st.subheader("Future of Shoethentic")
    st.markdown("We plan to improve the accuracy of the AI model even more when checking for shoes and integrate it into the Sire website later on.")


