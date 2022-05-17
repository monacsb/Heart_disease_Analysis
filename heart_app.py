# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:37:35 2022

@author: Mona
"""

import os
import numpy as np
import pickle
import streamlit as st

#%%PATH

MODEL_PATH = os.path.join(os.getcwd(),'trained_model.sav')

#%%
#Load the model

loaded_model = pickle.load(open(MODEL_PATH,'rb'))

#%%Creating a function

def heart_disease(input_data):
    input_data_array=np.array(input_data)
    
    #reshape the array as we are predicting for one instance
    input_data_reshaped=input_data_array.reshape(1,-1)
    
    
    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    if (prediction[0] == 0):
        return st.success('You are safe from heart disease')
    else:
        return st.error('You have heart disease')
    
def main():
    
    #giving a title
    st.title('Heart Disease Prediction App')

    age = st.text_input('Enter your age:')
    
    sex = st.radio("Your gender:",('1','0'))
    if (sex == '1'):
        st.info('Male')
    else:
        st.info('Female')
 
    # st.caption('Please enter your gender in value 1 for Male, 2 for Female')
    
    cp = st.text_input('Enter Chest Pain type:0=Typical_angina,1=Atypical_angina,2=Non_Anginal_pain,3=Asymptomatic')

    trtbps = st.text_input('Resting blood pressure in mm Hg')
    
    chol = st.text_input('Cholestrol in mg/dl')
    
    fbs = st.text_input('Fasting blood sugar > 120mg/dl 1:True 0:False')
    
    restecg = st.text_input('Resting electrocardiographic results:0=normal,1=having STT wave abnormality,2=showing probable or definite left ventricular hypertrophy by Estes criteria')
                       
    thalachh = st.text_input('Enter your maximum heart rate achieved')
    
    exng = st.text_input('Exercise induced angina:1 = yes, 0 = no')
    
    oldpeak = st.text_input('Enter your old peak')
    
    slp = st.text_input('Enter your slope: 1=upsloping, 2=flat, 3=downsloping')
    
    caa = st.text_input('Number of major vessels 0 to 3')
    
    thall = st.text_input('Maximum heart rate achieved')
    
    #code for prediction
    diagnosis=''
    
    #Prediction button
    if st.button('Heart Disease Test Result'):
        diagnosis = heart_disease([age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall])
    st.success(diagnosis)
 
#this line run the main function in cmd prompt
if __name__ == '__main__':
    main()