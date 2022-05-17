# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:37:35 2022

@author: Mona
"""

import os
import numpy as np
import pickle


#%%PATH

MODEL_PATH = os.path.join(os.getcwd(),'trained_model.sav')

#%%
#Load the model

loaded_model = pickle.load(open(MODEL_PATH,'rb'))

#%%test deployment

input_data=(67,1,0,160,286,0,0,108,1,1.5,1,3,2)

input_data_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_array.reshape(1,-1)


prediction=loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print("You are safe from heart disease")
else:
        print("You have heart disease")