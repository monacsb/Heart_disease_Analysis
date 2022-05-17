# Heart_disease_Analysis
This project analyze a dataset consists of patient's records that are vulnerable to heart attack and non-heart attack. 

# Description
"According to World Health Organisation (WHO), every year around 17.9 million deaths are due to cardiovascular diseases (CVDs) predisposing CVD becoming the leading cause of death globally.CVDs are a group of disorders of the heart and blood vessels, if left untreated it may cause heart attack. Heart attack occurs due to the presence of obstruction of blood flow into the heart. The presence of blockage may be due to the accumulation of fat, cholesterol, and other substances. Despite treatment has improved over the years and most CVD’s pathophysiology have been elucidated, heart attack can still be fatal." 

As many scientists and researcher working on the prevention, as a data analyst we can do a little help too. 

This project basically gives you a prediction on dataset by developing a reliable model using a machine learning approach. 
In this case, Logistic Regression is used in this project. 

To fine tune on our data, initial steps called Exploratory Data Analysis has been conducted. <br />
This also, by understanding the insights of our data such by looking at 
**type of our data:** <br />
**df.info()**
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

**Checking our null values**
There is no null values contain in the dataset
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
Unfortunately, there is one duplicate data. 
In order to have a quick glance on exact duplicate row, we can extract that specific row(s) <br />
In this case, we only have one. <br />
 age  sex  cp  trtbps  chol  fbs  ...  exng  oldpeak  slp  caa  thall  output
164   38    1   2     138   175    0  ...     0      0.0    2    4      2       1

**Action taken: to drop this duplicate values.
**
Our data set shows 164 persons with heart disease and 138 without heart disease. 

1 165
0 138

This shows our dataset is balance. 

The image below is observation on correlation using heatmap<br/>
Found a lowest correlation with target variable are fbs and chol. Whereas, the rest variables showing significant correlation between <br/>
target variable.

![image](https://user-images.githubusercontent.com/103228610/168871223-6e071492-8422-4121-9ed3-b1461c80ab7f.png)


# How to use it 
Clone repo and run it. <br />
heart_analysis.py is a script to train the data. <br />
heart_deploy.py is a script for deployment. This file gives you prediction on one instance. <br />

# Requirement
Spyder 
Python 3.8
Windows 10 or even latest version
Anaconda prompt in order to execute our tensorflow environment

# Results
The accuracy score on training data is 84%  whereas accuracy score on training data is 79%. Therefore, this model is showing a good model as it has no overfitting or underfitting.
<img width="361" alt="image" src="https://user-images.githubusercontent.com/103228610/168871883-039e0d0f-7d6b-4a0c-a62e-ca03d1be526d.png">


# Model
Logistic Regression 
Model is saved under trained_model.sav using pickle in order for deployment. 

# Streamlit App
Model is deployed and performed prediction on a web app using Streamlit.

# Credits
Thanks to rashikrahmanpritom for the data
https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
