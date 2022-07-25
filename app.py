#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 12:36:32 2022

@author: angela
"""

#%%

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import pickle 
import os

#%%

MODEL_PATH = os.path.join(os.getcwd(),'model','model.pkl')
with open(MODEL_PATH,'rb') as file:
    classifier = pickle.load(file)


#%% 

IMAGE_PATH = os.path.join(os.getcwd(),'Static','Header.jpg')
IMAGE_PATH1 = os.path.join(os.getcwd(),'Static','risk-factors.png')


image = Image.open(IMAGE_PATH)
st.image(image, use_column_width=True)
st.title("Cardiovascular Disease Prediction")
st.header("This is an app to predict whether or not you have chance of cardiovascular disease")
st.subheader("What you need to know about Cardiovascular Disease?")
st.write("Cardiovascular diseases (CVDs) are a group of disorders of the heart and blood vessels. They include: coronary heart disease – a disease of the blood vessels supplying the heart muscle;cerebrovascular disease – a disease of the blood vessels supplying the brain;peripheral arterial disease – a disease of blood vessels supplying the arms and legs;rheumatic heart disease – damage to the heart muscle and heart valves from rheumatic fever, caused by streptococcal bacteria;congenital heart disease – birth defects that affect the normal development and functioning of the heart caused by malformations of the heart structure from birth; and deep vein thrombosis and pulmonary embolism – blood clots in the leg veins, which can dislodge and move to the heart and lungs. Heart attacks and strokes are usually acute events and are mainly caused by a blockage that prevents blood from flowing to the heart or brain. The most common reason for this is a build-up of fatty deposits on the inner walls of the blood vessels that supply the heart or brain. Strokes can be caused by bleeding from a blood vessel in the brain or from blood clots.")

st.subheader("Risk Factor of Cardiovascular Disease")
image = Image.open(IMAGE_PATH1)
st.image(image, use_column_width=True)

st.sidebar.header('Please fill in the details')
with st.sidebar:
    with st.form(key='my_form'):
        age = st.sidebar.number_input('Age')
        cp = st.sidebar.selectbox('Chest Pain Type ( 0 = asymptomatic ; 1 = typical angina; 2 = atypical angina; 3 = non-anginal pain)',(0,1,2,3))  
        trtbps = st.sidebar.number_input('Diastolic Pressure (mm/Hg)')
        chol = st.sidebar.number_input('Cholesterol reading (mg/dl)')
        thalachh = st.sidebar.number_input('Maximum heart rate')
        exng = st.sidebar.selectbox('Exercise Induced Angine (0 = no; 1 = yes)',(0,1)) 
        oldpeak = st.sidebar.number_input('ST depression')
        caa = st.sidebar.selectbox('Number of major vassels by fluoroscopy(0,1,2,3)',(0,1,2,3)) 
        thall = st.sidebar.selectbox('Thalassemia(0 = null,1 = fixed defect,2 = normal,3 = reversable defect)',(0,1,2,3)) 
        submitted = st.sidebar.button("Submit")


st.subheader("Result")
if submitted:
        new_data = np.expand_dims([age,cp,trtbps,chol,thalachh,exng,oldpeak,caa,thall],axis=0)
        outcome = classifier.predict(new_data)[0]
        if outcome == 0:
            st.success('No Cardiovascular disease detected')
            st.write('✨Keep it up!✨')
            st.balloons()
        else:
            st.warning('Cardiovascular diseas detected')
            st.write('Please consult doctor for further information.')
            st.snow()




