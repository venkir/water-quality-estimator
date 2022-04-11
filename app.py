#!/usr/bin/env python
# coding: utf-8

# ## Korattur Lake Water Quality Prediction Application using PyCaret and StreamLit
# 
# #### By: Venki Ramachandran
# #### Dated: 01-Apr-2022
# 
# **Description & Credits:** The dataset is from a Lake in Chennai, India and it was staged in github by Jahnavi Srividya. It can be found at https://github.com/JahnaviSrividya/Korattur-Lake-Water-Quality-Dataset. I followed the methodology outlined in the paper that can be found here: https://www.sciencedirect.com/science/article/abs/pii/S0013935121010148
# 
# This dataset has been sourced from the Korattur Lake that is located in Chennai, a south Indian metropolis. Deemed the largest it spans over 990 acres, supplying public drinking water for over eighteen years. The dataset contains observations of a ten-year period, ranging from 2019 to 2019. About 5000 records under 9 parameters are present. The parameters are Turbidity, TDS, COD, PH, Phosphate, Iron, Nitrate, Sodium and Chloride. The 5000 records broadly fall into two types, namely the binary and multi-class.
# 
# **Methodology:**
# 1. Load Data. Do EDA. 
# 2. Remove Outliers, Normalize the data and use SMOTE to make sure the two class predictor is balanced.
# 3. I plan to use Random Forest, K Nearest Beighbor, Naive Bayes and get some results. Measure accuracy. Pick the best one from the above
# 4. Deploy the above using Streamlit so that external users can connect and test the data.
# 
# 
# Load all the libraries needed
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load the Dataset
df = pd.read_csv("Korattur-Lake-Water-Quality-BinaryClass-Dataset.csv")
# Double Check
df.isna().sum()
# Rename the predictor column
df = df.rename({'Class': 'Potability'}, axis=1)  # new method
df = df.rename({'COD(mg/L)': 'COD-mg/L'}, axis=1)  # new method

#### Normalization
data = df.copy(deep=True)

def do_normalization(df, col):
    df[col] = [ (i-df[col].min())/(df[col].max()-df[col].min()) for i in df[col]]

columns_to_normalize = ['pH', 'TDS', 'Turbidity', 'Phospate', 'Iron', 'Chlorine',
       'COD-mg/L', 'Sodium', 'Nitrate']

for column in columns_to_normalize:
    do_normalization(data, column)


# SMOTE to fix imbalance. There are a lot of rows with Potability set = 0 and a very few set = 1
# SMOTE (synthetic minority oversampling technique) is one of the most commonly used oversampling methods to solve the 
# imbalance problem. It aims to balance class distribution by randomly increasing minority class examples by replicating them. 
# SMOTE synthesises new minority instances between existing minority instances.
#Installing imblearn
#get_ipython().system('pip install -U imbalanced-learn')
#Importing SMOTE
import imblearn
from imblearn.over_sampling import SMOTE
#Oversampling the data
smote = SMOTE(random_state = 101)
X, y = smote.fit_resample(data[['pH', 'TDS', 'Turbidity', 'Phospate', 'Iron', 'Chlorine',
       'COD-mg/L', 'Sodium', 'Nitrate']], data['Potability'])

#Same number of rows = 8638
# Creating a new Oversampling Data Frame
df_oversampler = pd.DataFrame(X, columns = ['pH', 'TDS', 'Turbidity', 'Phospate', 'Iron', 'Chlorine',
       'COD-mg/L', 'Sodium', 'Nitrate'])
df_oversampler['Potability'] = y
sns.countplot(df_oversampler['Potability'])

# Let us try a K Nearest Neighbors Classifier
# Split the dataset into X and Y
X = df_oversampler.iloc[:, :-1].values
y = df_oversampler.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state=1)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k
knn.fit(X_train,y_train)
prediction = knn.predict(X_test)
#print(" {} nn score: {} ".format(3,knn.score(X_test,y_test)))


# ### We got a 91.9% accuracy with 3 Nearest Neighbor Classifier

# Let us try to see what the ideal value of K should be
score_list = []
for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(X_train,y_train)
    score_list.append(knn2.score(X_test,y_test))

# K == 2 seems to be the best value

# Now with two (2) neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2) # n_neighbors = k
knn.fit(X_train,y_train)
prediction = knn.predict(X_test)
#print(" {} nn score: {} ".format(3,knn.score(X_test,y_test)))


# ### The accuracy increased to 95.75%

# Naive Bayes
# training the NB model and making predictions
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()

# fit
mnb.fit(X_train,y_train)

# predict class
y_pred_class = mnb.predict(X_test)

# predict probabilities
y_pred_proba = mnb.predict_proba(X_test)

# printing the overall accuracy
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)

# confusion matrix
confusion = metrics.confusion_matrix(y_test, y_pred_class)
#print(confusion)

TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
TP = confusion[1, 1]

sensitivity = TP / float(FN + TP)
##print("sensitivity",sensitivity)


specificity = TN / float(TN + FP)
#print("specificity",specificity)


precision = TP / float(TP + FP)
#print("precision ", metrics.precision_score(y_test, y_pred_class))
#print("PRECISION SCORE :",metrics.precision_score(y_test, y_pred_class))
#print("RECALL SCORE :", metrics.recall_score(y_test, y_pred_class))
##print("F1 SCORE :",metrics.f1_score(y_test, y_pred_class))

# creating an ROC curve
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_proba[:,1])
roc_auc = auc(false_positive_rate, true_positive_rate)

# area under the curve
#print (roc_auc)

# ### Random Forest Classifier gave the best accuracy we will use that for the app demo

# Split the dataset into X and Y
X = df_oversampler.iloc[:, :-1].values
y = df_oversampler.iloc[:, -1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Let us use a Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
#print(cm)
#print("the accuracy is",accuracy_score(y_test, y_pred))


# ### We got a 99.9% accuracy with Random Forest Classifier
import os
import joblib
# save
joblib.dump(classifier, "./random_forest.joblib")
# load the saved model
# no need to initialize the loaded_rf
loaded_rf_classifier = joblib.load("./random_forest.joblib")
def predict_quality(model, df):
    
    predictions_data = model.predict(df)
    #print("prediction = ", predictions_data)
    return predictions_data[0]

import streamlit as st

# Streamlit Code
st.title('Water Quality Prediction Classifier')
st.subheader('A Random Forest Classifier to predict Water Quality based on sensor data!!')
st.write(
    "This is a web app to classify the quality of water as Potable or 'Not Potable' based on\n"
    "several features that you can see in the sidebar. Please adjust the\n"
    "value of each feature. After that, click on the Predict button at the bottom to\n"
    "see the prediction of the classifier.\n\n"
    
    "Developed By: Venki Ramachandran on 12-March-2022."
)


# Next, we need to let the user to specify the value of our features. Since our features are all numeric features, it will be best to represent them with a slider widget. To create a slider widget, we can use slider() function from Streamlit
# Allow the user to select new features for prediction
pH = st.sidebar.slider(label = 'pH Value', min_value = 7.5,
                          max_value = 7.6 ,
                          value = 7.55,
                          step = 0.01)

TDS = st.sidebar.slider(label = 'TDS', min_value = 700,
                          max_value = 950 ,
                          value = 800,
                          step = 10)
                          
Turbidity = st.sidebar.slider(label = 'Turbidity', min_value = 1,
                          max_value = 3 ,
                          value = 2,
                          step = 1)                          

Phospate = st.sidebar.slider(label = 'Phospate', min_value = 0.005,
                          max_value = 0.025 ,
                          value = 0.015,
                          step = 0.005)

Iron = st.sidebar.slider(label = 'Iron', min_value = 0.32,
                          max_value = 0.4 ,
                          value = 0.35,
                          step = 0.01)
   
Chlorine = st.sidebar.slider(label = 'Chlorine', min_value = 2,
                          max_value = 10,
                          value = 6,
                          step = 1)

COD = st.sidebar.slider(label = 'COD-mg/L', min_value = 300,
                          max_value = 420 ,
                          value = 350,
                          step = 1)

Sodium = st.sidebar.slider(label = 'Sodium', min_value = 4,
                          max_value = 15,
                          value = 10,
                          step = 1)

Nitrate = st.sidebar.slider(label = 'Nitrate', min_value = 5,
                          max_value = 8,
                          value = 6,
                          step = 1)
 


# Next we need to convert all of those user input values into a dataframe. Then, we can use the dataframe as the input of our model’s prediction
features = {
            'pH Value': pH, 'TDS': TDS,
            'Turbidity': Turbidity, 'Phospate': Phospate,
            'Iron': Iron, 'Chlorine': Chlorine,
            'COD-mg/L': COD, 'Sodium': Sodium, 'Nitrate': Nitrate
            }
 

features_df  = pd.DataFrame([features])

st.table(features_df)  

if st.button('Predict'):
    
    prediction = predict_quality(loaded_rf_classifier, features_df)
    if prediction == 0:
        res = 'Potable'
    else:
        res = 'Not Drinkable'
    
    st.write(' Based on feature values, your water quality is '+ str(res))