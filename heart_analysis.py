# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:32:51 2022

@author: Mona
"""

import pandas as pd
import os
import numpy as np

#To visualize
import matplotlib.pyplot as plt
import seaborn as sns

#features selection
from sklearn.preprocessing import StandardScaler 

#Data processing
from sklearn.model_selection import train_test_split

#Building model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

#Reporting
from sklearn.metrics import confusion_matrix, classification_report

#Pickle
import pickle
#%%STATIC
DATA_PATH = os.path.join(os.getcwd(),'heart.csv')
SCALER_PATH = os.path.join(os.getcwd(),'scaler.pkl')
MODEL_PATH = os.path.join(os.getcwd(),'trained_model.pkl')
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

#%% save scaler 
pickle.dump(sc,open(SCALER_PATH,'wb'))

#%%
X = scaled_feature
y = df['output']

#%% Train test split

X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,
                                                   stratify=y,random_state=42)

#The original data contains 302 dataset, out of that 80% is used for training
#the dataset and 20% as test data 
print(X.shape,X_train.shape,X_test.shape)

#%% Step 5:Building Model

#1: Training the model

#Define model
knn = [('KNN Classifier', KNeighborsClassifier())]
tree = [('Decision Tree Classifier', DecisionTreeClassifier())]
rf = [('Random Forest Classifier', RandomForestClassifier())]
lr = [('Logistic Regression',LogisticRegression())]

#load the model into pipeline
pipeline_knn = Pipeline(knn)
pipeline_tree = Pipeline(tree)
pipeline_rf = Pipeline(rf)
pipeline_lr = Pipeline(lr)

#create a list to store all the pipelines
pipelines = [pipeline_knn, pipeline_tree, pipeline_rf,pipeline_lr]

#fit the training data into the pipelines
for pipe in pipelines:
    pipe.fit(X_train,y_train)

#checking the accuracy score of these models
pipe_dict = {0:'KNeighbors Classifier', 1:'Decision Tree Classifier',
             2:'Random Forest Classifier', 3:'Logistic Regression'}

#prediction of models
prediction = []
best_score = 0
best_scaler = 0
best_pipeline =''

for i, model in enumerate(pipelines):
    prediction.append(model.predict(X_test))
    print('{} Test Accuracy:{}'.format(pipe_dict[i],
                                       model.score(X_test,y_test)))
    if model.score(X_test, y_test) > best_score:
        best_score = model.score(X_test, y_test)
        best_scaler = i
        best_pipeline = model

print('Best model is {} with accuract of {}%'.format(pipe_dict[best_scaler],(best_score)*100))

#%% So based on the best model, the report generated below
best_pipeline_pred = prediction[2]
print(classification_report(y_test, best_pipeline_pred))
print(confusion_matrix(y_test,best_pipeline_pred))

#%%Save model

with open(MODEL_PATH,'wb') as file:
    pickle.dump(best_pipeline,file)

