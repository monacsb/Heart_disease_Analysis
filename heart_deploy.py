# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:37:35 2022

@author: Mona
"""

import os
import numpy as np
import pickle


#%%PATH

MODEL_PATH = os.path.join(os.getcwd(),'trained_model.pkl')
SCALER_PATH = os.path.join(os.getcwd(),'scaler.pkl')

#%%
#Load the model

sc = pickle.load(open(SCALER_PATH,'rb'))
model=pickle.load(open(MODEL_PATH,'rb'))

#%%test deployment

input_data = (57,1,2,128,229,0,0,150,0,0.4,1,1,3)

input_data_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_array.reshape(1,-1)

#scale the patient info 
std_data = sc.transform(input_data_reshaped)
print(std_data)

prediction=model.predict(std_data)
print(prediction)

if (prediction[0] == 0):
    print("You are safe from heart disease")
else:
        print("You have heart disease")