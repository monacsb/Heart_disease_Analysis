# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:32:51 2022

@author: Mona
"""

import pandas as pd
import os
import numpy as np
import pickle

#To visualize
import matplotlib.pyplot as plt
import seaborn as sns

#features selection
from sklearn.preprocessing import StandardScaler 

#Data processing
from sklearn.model_selection import train_test_split

#Building model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

#Pickle
import pickle
#%%
DATA_PATH = os.path.join(os.getcwd(),'heart.csv')
#%% EDA

#Step 1: Load Data

df = pd.read_csv(DATA_PATH)

#%%
#Step 2: Data Inspection

df.info()

#to check the outliers
df.describe()
df.describe().T

#checking null values
#there is no null values contain in the dataset
df.isnull().sum()

#checking on duplicate values 
#found one duplicate value
df.duplicated().sum()

#extract the duplicate rows
df.loc[df.duplicated(),:]

#drop duplicate values
df = pd.DataFrame(df).drop_duplicates()

#The data set shows 164 person with heart disease and 138 without heart disease
#It is balanced dataset
df.output.value_counts()

#Correlation matrix
#Correlation observed using heatmap
corr=df.corr()
corr['output'].sort_values(ascending=False)

#observing corr using heatmap
#The lowest correlation with target variable are fbs and chol
# Whereas the rest variables showing significant correlation between 
#target variable 
plt.figure(figsize=(15,10))
sns.heatmap(corr, annot=True, cmap='Reds')
plt.show()

#%%
#Step 3: Data Cleaning

#%%
#Step 4: Features Selection

X = df.drop(labels=['output'],axis=1)
y = df['output']

#%% Data Standardization

#scaling the data
#define scaler
sc = StandardScaler()

#fit and transform scaler on the training dataset
sc.fit(X)

scaled_feature = sc.transform(X)

#now our values has been scaled into standard format 
print(scaled_feature)

#%%
X = scaled_feature
y = df['output']

#%% Train test split

X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,
                                                   stratify=y,random_state=2)

#The original data contains 302 dataset, out of that 80% is used for training
#the dataset and 20% as test data 
print(X.shape,X_train.shape,X_test.shape)

#%% Step 5:Building Model

#1: Training the model

#Define model
model = LogisticRegression()
model.fit(X_train,y_train)

#2: Model evaluation
#Accuracy score on training data
X_train_prediction = model.predict(X_train)
trained_accuracy = accuracy_score(X_train_prediction, y_train)
print('Accuracy score of the training data: ',trained_accuracy)

#The accuracy score is 84%
#%%
#Accuracy score on training data
X_test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(X_test_prediction, y_test)
print('Accuracy score of the test data: ',test_accuracy)

#the accuracy score is 79%

#%% 
#The train score and test score shows that the model is not over fitted.

#%%test deployment

# input_data = (61,1,0,140,207,0,0,138,1,1.9,2,1,3)

# input_data_array = np.asarray(input_data)

# #reshape the array as we are predicting for one instance
# input_data_reshaped = input_data_array.reshape(1,-1)

# #scale the patient info 
# std_data = sc.transform(input_data_reshaped)
# print(std_data)

# prediction=model.predict(std_data)
# print(prediction)

# if (prediction[0] == 0):
#     print("You are safe from heart disease")
# else:
#         print("You have heart disease")
#%%Save model

filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))
MODEL_PATH = os.path.join(os.getcwd(),'trained_model.sav')
