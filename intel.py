#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import streamlit as st
import cv2
from keras.models import load_model


# In[2]:


model = load_model('E:\\Python\\intel_image.h5')


# In[ ]:


CLASS_NAMES = ['builidings','Forest','glacier','Mountain','Sea','Street']


# In[3]:


st.title("Dog Breed Prediction")
st.markdown("Upload an image of the dog")


# In[ ]:


dog_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict')
if submit:


    if dog_image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        # Displaying the image
        st.image(opencv_image, channels="BGR")
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (150,150))
        #Convert image to 4 Dimension
        opencv_image.shape = (1,150,150,3)
        #Make Prediction
        Y_pred = model.predict(opencv_image)

        st.title(str("The Image is "+CLASS_NAMES[np.argmax(Y_pred)]))


# In[ ]:




