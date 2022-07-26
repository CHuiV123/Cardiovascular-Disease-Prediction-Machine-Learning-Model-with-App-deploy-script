# Cardiovascular-Disease-Prediction-Machine-Learning-Model-with-App-deploy-script

## Project Description 
This is a machine learning model to predict whether a person has cardiovascular disease based on several criterion. An application based on the the best predicting model is developed using Streamlit after performing a sequence of steps from Exploratory Data Analysis(EDA), features selection to model training. 
  
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
 ![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
 ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
 ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
 ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)

# How to Install and Run the Project 
If you wish to run this model on your pc, you may download "heart.csv" (in dataset folder) together with Heart_attack_training.py and app.py file from the depository section. The selected machine learning models for application development has been uploaded for you as "model.pkl" in under the folder named model. 

To view my application on webpage using local host: Open terminal from your pc > streamlit run " THE LOCATION PATH WHERE U SAVE THE app.py FILE" > Enter > wait for the local host address to appear > click / copy paste the local host address to your web browser 


Software required: Spyder, Python(preferably the latest version) 

Additional modules needed: Sklearn, Streamlit 



# Processing Parameter 
Based on my machine learning model, the best performing combination of scaler and classifier is MinMaxScaler+ Gradient Boost Classifier with a score of 0.824. 

![alt text](https://github.com/CHuiV123/Cardiovascular-Disease-Prediction-Machine-Learning-Model-with-App-deploy-script/blob/c5c0265fc74c836c7e0dfbad018f5e1183abbbd4/static/Best%20pipe.png)

After performing Gridsearch CV hyperparameter tuning, below is the params that yield the best result.  
![alt text](https://github.com/CHuiV123/Cardiovascular-Disease-Prediction-Machine-Learning-Model-with-App-deploy-script/blob/c5c0265fc74c836c7e0dfbad018f5e1183abbbd4/static/hyperparameter.png)

# Classification Report 
![alt text](https://github.com/CHuiV123/Cardiovascular-Disease-Prediction-Machine-Learning-Model-with-App-deploy-script/blob/c5c0265fc74c836c7e0dfbad018f5e1183abbbd4/static/classification%20report.png)

# Streamlit Application View
![alt text](https://github.com/CHuiV123/Cardiovascular-Disease-Prediction-Machine-Learning-Model-with-App-deploy-script/blob/9018cd0a1c85c783048c67d28221acc2b053c617/static/Application%20screen%20shot.png)


***
# Results and Finding 
![alt text](https://github.com/CHuiV123/Cardiovascular-Disease-Prediction-Machine-Learning-Model-with-App-deploy-script/blob/8b6389c9ce65c188bee021d9a27a3451747f7db1/static/App%20Testing%20Dataset.png)

By using the test data set above, 9 out of 10 were predicted correctly based on my machine learning model. The application developed from my machine learning model has an accuracy of 90%**. 

The training dataset consists of a total of 303 data entries. However, imbalance dataset were spotted and this may affect the accuracy performance of the machine learning model. 
Outliers were detected in a few columns, but were not removed after the verification from valid source** whereby the outliers value is still within acceptable range. 


**
Accuracy of 90%: if more data were collected and input into the model for model retraining purposes, accuracy may have further improvement. 

** 
National Library of Medicine for highest diastolic reading cross checking. 
https://pubmed.ncbi.nlm.nih.gov/7741618/  

Guiness World Record for higest possible cholesterol level cross checking. 
https://www.guinnessworldrecords.com/world-records/highest-triglyceride-level

## Credits
This datasets is provided by [Kaggle] [Rashik Rahman] [https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset] 
