#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:07:29 2022

@author: angela
"""

#%% Imports 

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import scipy.stats as ss 
import seaborn as sns
import pandas as pd 
import numpy as np 
import pickle
import os 

#%% Constants 

CSV_PATH = os.path.join(os.getcwd(),'dataset','heart.csv')

MODEL_PATH = os.path.join(os.getcwd(),'model','model.pkl')


#%% 

def plot_con_graph(con,df): 
    '''
    ..... this function is meant to plot continuous data using seaborn function 

    Parameters
    ----------
    con : continuous list
        DESCRIPTION.
    df : dataframe
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    for i in con: 
        plt.figure()
        sns.distplot(df[i])  # for continuous use distribution plot 
        plt.show()



def plot_cat_graph(cat,df):
    """
    ..... this function is meant to plot categorical data using seaborn function

    Parameters
    ----------
    cat : category list
        DESCRIPTION.
    df : dataframe
        DESCRIPTION.

    Returns
    -------
    None.

    """
    for i in cat: 
        plt.figure()
        sns.countplot(df[i])  # for categorical use count plot 
        plt.show()



def cramers_corrected_stat(confusion_matrix):
   """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
   """
   chi2 = ss.chi2_contingency(confusion_matrix)[0]
   n = confusion_matrix.sum()
   phi2 = chi2/n
   r,k = confusion_matrix.shape
   phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
   rcorr = r - ((r-1)**2)/(n-1)
   kcorr = k - ((k-1)**2)/(n-1)
   return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


#%% 1) Data Loading 

df = pd.read_csv(CSV_PATH)

#%% 2) Data Inspection 

df.info()  # total 303 entries for this dataset 
df.describe().T 
df.isna().sum()
# no NaNs value in dataset, but spotted out of range index entries in thal and caa column
df.duplicated().sum() # there is 1 duplicated entry

plt.figure(figsize=(18,12))
df.boxplot() # boxplot shows that trtbps, chol, thalachh has outliers 
df.columns

con = ['age','trtbps','chol','thalachh','oldpeak']

cat = list(df.drop(con,axis=1))

plot_con_graph(con, df)
plot_cat_graph(cat, df)

# This is imbalance dataset 


#%% 3) Data Cleaning 

# to filter out of range index, change it to NaNs 
df['thall'] = df['thall'].replace(0,np.nan)
df['caa'] = df['caa'].replace(4,np.nan)

df.isnull().sum()


# to impute data for out of range index in thall and caa column 
df['thall'] = df['thall'].fillna(df['thall'].mode()[0])
df['caa'] = df['caa'].fillna(df['caa'].mode()[0])

# outliers 
# the highest diastolic ever recorded according to National library of medicine was 360(refer link below), so value of 200 trtbps in this case will be regard as acceptable record 
# https://pubmed.ncbi.nlm.nih.gov/7741618/  

# the highest cholesterol level ever recorded was 3165mg/dl and it is recorded in the guiness world record. refer link below. so outlier of >500 mg/dl here is consider valid data entry. it may be exceptional case of chronic disease patient 
# https://www.guinnessworldrecords.com/world-records/highest-triglyceride-level

# dropping duplicate 
df = df.drop_duplicates()
df.duplicated().sum()

#%% 4) Features Selection 

# Target : Output (Categorical) Whether or not one has cardiovascular disease  

# cat vs cat 
#cramer's V 

for i in cat:
    print(i)
    confusion_matrix = pd.crosstab(df[i],df['output']).to_numpy()
    print(cramers_corrected_stat(confusion_matrix))

# cp,exng, caa and thall is showing some correlation to the output. thall has the highest ratio which is 0.521

for i in con:
    print(i)
    lr=LogisticRegression()
    lr.fit(np.expand_dims(df[i],axis=-1),df['output'])
    print(lr.score(np.expand_dims(df[i],axis=-1),df['output']))
# all of the continuous data columns has high correlation with the output 

#%% 5) Data preprocessing 

X = df.loc[:,['age','restecg','trtbps','chol','thalachh','exng','oldpeak',
              'caa','thall']]
y = df['output']


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,
                                                     random_state=123)

#%%Model-development ---> pipeline


pipeline_mms_lr = Pipeline([
                            ('Min_Max_Scaler',MinMaxScaler()),
                            ('Logistic_Classifier',LogisticRegression())
                            ]) # Pipeline([STEPS])

pipeline_ss_lr = Pipeline([
                            ('Standard_Scaler',StandardScaler()),
                            ('Logistic_Classifier',LogisticRegression())
                            ]) # Pipeline([STEPS])

# Decision Tree
pipeline_mms_dt = Pipeline([
                            ('Min_Max_Scaler',MinMaxScaler()),
                            ('Tree_Classifier',DecisionTreeClassifier())
                            ]) # Pipeline([STEPS])

pipeline_ss_dt = Pipeline([
                            ('Standard_Scaler',StandardScaler()),
                            ('Tree_Classifier',DecisionTreeClassifier())
                            ]) # Pipeline([STEPS])

# Random Forest
pipeline_mms_rf = Pipeline([
                            ('Min_Max_Scaler',MinMaxScaler()),
                            ('Forest_Classifier',RandomForestClassifier())
                            ]) # Pipeline([STEPS])

pipeline_ss_rf = Pipeline([
                            ('Standard_Scaler',StandardScaler()),
                            ('Forest_Classifier',RandomForestClassifier())
                            ]) # Pipeline([STEPS])

# SVM
pipeline_mms_svc = Pipeline([
                            ('Min_Max_Scaler',MinMaxScaler()),
                            ('SVM_Classifier',SVC())
                            ]) # Pipeline([STEPS])

pipeline_ss_svc = Pipeline([
                            ('Standard_Scaler',StandardScaler()),
                            ('SVM_Classifier',SVC())
                            ]) # Pipeline([STEPS])

# KNN
pipeline_mms_knn = Pipeline([
                            ('Min_Max_Scaler',MinMaxScaler()),
                            ('KNN_Classifier',KNeighborsClassifier())
                            ]) # Pipeline([STEPS])

pipeline_ss_knn = Pipeline([
                            ('Standard_Scaler',StandardScaler()),
                            ('KNN_Classifier',KNeighborsClassifier())
                            ]) # Pipeline([STEPS])

# GBoost
pipeline_mms_gb = Pipeline([
                            ('Min_Max_Scaler',MinMaxScaler()),
                            ('GBoost_Classifier',GradientBoostingClassifier())
                            ]) # Pipeline([STEPS])

pipeline_ss_gb = Pipeline([
                            ('Standard_Scaler',StandardScaler()),
                            ('GBoost_Classifier',GradientBoostingClassifier())
                            ]) # Pipeline([STEPS])


# To create A List To Store All The Pipeline
pipelines = [pipeline_mms_lr, pipeline_ss_lr,pipeline_mms_dt,pipeline_ss_dt,
             pipeline_mms_rf,pipeline_ss_rf,pipeline_mms_svc,pipeline_ss_svc,
             pipeline_mms_knn,pipeline_ss_knn,pipeline_mms_gb,pipeline_ss_gb]

for pipe in pipelines:
    pipe.fit(X_train,y_train)




pipelines = [pipeline_mms_lr, pipeline_ss_lr,pipeline_mms_dt,pipeline_ss_dt,
             pipeline_mms_rf,pipeline_ss_rf,pipeline_mms_knn,pipeline_ss_knn,
             pipeline_mms_gb,pipeline_ss_gb,pipeline_mms_svc,pipeline_ss_svc]

for pipe in pipelines:
    pipe.fit(X_train,y_train)

pipe_score = []

for i,pipe in enumerate(pipelines):
    pipe_score.append(pipe.score(X_test,y_test))

print(pipelines[np.argmax(pipe_score)])
print(pipe_score[np.argmax(pipe_score)])

best_pipe = pipelines[np.argmax(pipe_score)]


#%% Gridsearch cv
# this is to search the best combination from best model


pipeline_mms_gb = Pipeline([
    ('Min_Max_Scaler', MinMaxScaler()),
    ('gb_Classifier', GradientBoostingClassifier())
    ]) #Pipeline([STEPS])


grid_param = [{'gb_Classifier__n_estimators':[100,200,500],
               'gb_Classifier__learning_rate':[0.01,0.05,0.1],
               'gb_Classifier__max_depth':[1,2,3]
               }]


gridsearch = GridSearchCV(pipeline_mms_gb,grid_param,cv=5,verbose =1,n_jobs=1)
grid= gridsearch.fit(X_train,y_train)
gridsearch.score(X_test, y_test)
print(grid.best_params_)

best_model = grid.best_estimator_
#%% Model evaluation


y_true = y_test
y_pred = best_pipe.predict(X_test)

cr = classification_report(y_true,y_pred)

print(cr)


#%% model saving

with open(MODEL_PATH,'wb') as file:
    pickle.dump(best_model,file)


# Conclusion: Firstly, this dataset is imbalance and that may affect the 
# training accuracy of this machine learning model. 
# Second, the total data entries are consider low, more data should be provided 
# or included for the model to yield better perfomance result. 