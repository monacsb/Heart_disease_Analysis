# Heart_disease_Analysis
This project analyze a dataset consists of patient's records that are vulnerable to heart attack and non-heart attack. <br />

# Description
"According to World Health Organisation (WHO), every year around 17.9 million deaths are due to cardiovascular diseases (CVDs) predisposing CVD becoming the leading cause of death globally.CVDs are a group of disorders of the heart and blood vessels, if left untreated it may cause heart attack. Heart attack occurs due to the presence of obstruction of blood flow into the heart. The presence of blockage may be due to the accumulation of fat, cholesterol, and other substances. Despite treatment has improved over the years and most CVD’s pathophysiology have been elucidated, heart attack can still be fatal." <br />

As many scientists and researcher working on the prevention, as a data analyst we can do a little help too.  <br />

This project basically gives you a prediction on dataset by developing a reliable model using a machine learning approach. <br />
Logistic Regression is used as machine learning algorithm to predict the categorical dataset. <br />
So, we'll be looking at prediction of the output. In this case, logistic regression will be predicting the output of a categorical dependent variable. <br />

To fine tune on our data, initial steps called Exploratory Data Analysis has been conducted. <br />
This also, by understanding the insights of our data such by looking at <br />
**type of our data:** <br />
**df.info()** <br />
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 303 entries, 0 to 302
Data columns (total 14 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   age       303 non-null    int64  
 1   sex       303 non-null    int64  
 2   cp        303 non-null    int64  
 3   trtbps    303 non-null    int64  
 4   chol      303 non-null    int64  
 5   fbs       303 non-null    int64  
 6   restecg   303 non-null    int64  
 7   thalachh  303 non-null    int64  
 8   exng      303 non-null    int64  
 9   oldpeak   303 non-null    float64
 10  slp       303 non-null    int64  
 11  caa       303 non-null    int64  
 12  thall     303 non-null    int64  
 13  output    303 non-null    int64  
dtypes: float64(1), int64(13)

**Checking our null values** <br />
There is no null values contain in the dataset <br />
Out[57]: 
age         0
sex         0
cp          0
trtbps      0
chol        0
fbs         0
restecg     0
thalachh    0
exng        0
oldpeak     0
slp         0
caa         0
thall       0
output      0
dtype: int64

We can even check if our data consists of any duplicate values <br />
Unfortunately, there is one duplicate data. <br />
In order to have a quick glance on exact duplicate row, we can extract that specific row(s) <br />
In this case, we only have one. <br />
 <img width="466" alt="image" src="https://user-images.githubusercontent.com/103228610/168873794-a04b7112-541a-4823-b46f-1a2e3e618c97.png"> <br/>

**Action taken: to drop this duplicate values. <br />
**
Our data set shows 164 persons with heart disease and 138 without heart disease. <br />

<img width="56" alt="image" src="https://user-images.githubusercontent.com/103228610/168873664-45d32a32-97cd-49c3-bb45-b7b0f4d1a89f.png"> <br />

This shows our dataset is balance. <br />

The image below is observation on correlation using heatmap<br/>
Found a lowest correlation with target variable are fbs and chol. Whereas, the rest variables showing significant correlation between <br/>
target variable. <br />

![image](https://user-images.githubusercontent.com/103228610/168871223-6e071492-8422-4121-9ed3-b1461c80ab7f.png)


# How to use it  
Clone repo and run it. <br />
heart_analysis.py is a script to train the data. <br />
heart_deploy.py is a script for deployment. This file gives you prediction on one instance. <br />

# Requirement
Spyder <br />
Python 3.8 <br />
Windows 10 or even latest version <br />
Anaconda prompt in order to execute our tensorflow environment <br />

# Results 
I have used several model to train the data. The models will be dumped into a pipeline in order to extract the best model. Each time we train the model, it will gives us the best model. In this case, it gave me KNN Classifier with accuracy score of 80% <br/>
<img width="433" alt="best_model" src="https://user-images.githubusercontent.com/103228610/171986414-f473b091-56f9-4d01-b6cf-09a09eb48da0.png"> <br/>

Based on the best KNN model, I have generate the classification report to check on our f1-score <br/> 
<img width="378" alt="report_generation_score" src="https://user-images.githubusercontent.com/103228610/171986470-ad875e48-36b2-48d7-b32e-b90481902b8d.png"> <br/>


# Model
KNeighbors Classifier <br />
Decision Tree Classifier <br />
Random Forest Classifier <br />
Logistic Regression <br />

A list has been created to store all the model into the pipeline<br/>
Each time best model will be dumped into trained_model.pkl for deployment. <br />

# Deployment result
heart_deploy.py shows a simple test process by calling the input data manually whether a person have a heart disease or not<br/>


# Streamlit App
Model is deployed and performed prediction on a web app using Streamlit.<br />


# Credits
Thanks to rashikrahmanpritom for the data<br />
https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
