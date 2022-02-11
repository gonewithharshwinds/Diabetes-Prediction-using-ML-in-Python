# ===================================
# ===================================
# MIT License - Copyright (c) 2022 Harsh Mehta (gonewithharshwinds)
# ===================================
# Diabetes-Prediction-using-ML-Project-by-gonewithharshwinds
# ===================================
# Author : Harsh Mehta
# Date : Sat 12 Feb 2:50 am, 2022
# ===================================
# ===================================


# importing libraries

import numpy as np
import pandas as pn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score



# METHOD 1 : Using CSV
# Loading the diabetes dataset to pandas DataFrame
# diabetes_df = pn.read_csv('/Users/gonewithharshwinds/.../diabetes.csv')



# METHOD 2 : Using MySQL / other SQL db
from sqlalchemy import create_engine

# Credentials to database connection
hostname="localhost"
dbname="diabetes"
uname="root"
pwd="12345678"

# Create SQLAlchemy engine to connect to MySQL Database
engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
				.format(host=hostname, db=dbname, user=uname, pw=pwd))

# Convert dataframe to sql table if dataframe is designed here (not required because complete data is taken from external sql file.)                                  
# df.to_sql('users', engine, index=False)

# no of rows & columns in df
diabetes_df.shape
# (768,9)


diabetes_df.groupby('Outcome').mean()
# Separating data and labels
A = diabetes_df.drop(columns = 'Outcome', axis=1)
B = diabetes_df['Outcome']

# """Data Standardization"""
scaler = StandardScaler()
scaler.fit(A)
standardized_data = scaler.transform(A)
# update
A = standardized_data
B = diabetes_df['Outcome']



# Training -- Testing -- Splitting
A_training, A_testing, B_training, B_testing = train_test_split(A,B, test_size = 0.15, stratify = B, random_state = 2)

# Model Training
classifier = svm.SVC(kernel='linear')
# training the svm classifier
classifier.fit(A_training, B_training)


# Model Evaluation
# Accuracy Score
A_training_pred = classifier.predict(A_training)
training_data_acc = accuracy_score(A_training_pred, B_training)
print('Accuracy Score Obtained :', training_data_acc)
# I obtained " Accuracy Score Obtained : 0.7852760736196319 "



# Construct Predictive system
input_data = (5,111,76,0,0,37.6,0.290,30)

# changing the input data to numpy array

input_data_np_arr = np.asarray(input_data)

# reshape the array for predicting only 1 instance
input_data_reshaped = input_data_np_arr.reshape(1,-1)

# standardize this data

standard_data = scaler.transform(input_data_reshaped)
print(standard_data)
pred = classifier.predict(standard_data)
print(pred)

# print result obtained ==
# [[ 0.3429808  -0.30967058  0.35643175 -1.28821221 -0.69289057  0.71168975 -0.54928802 -0.27575966]]
# [0]

# The model has predicted correctly [0]

# for making interactive via terminal 
if (pred[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')
    
# result : The person is not diabetic




# Feeding positive diabetes data and executing
input_data2 = (5,166,70,19,175,25.4,0.588,50)

# changing the input data to numpy array

input_data_np_arr = np.asarray(input_data2)

# reshape the array for predicting only 1 instance
input_data_reshaped = input_data_np_arr.reshape(1,-1)

# standardize this data

standard_data = scaler.transform(input_data_reshaped)
print(standard_data)
pred = classifier.predict(standard_data)
print(pred)
if (pred[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')

# [[ 0.3429808   1.41167241  0.04624525 -0.09637905  0.82661621 -0.83672504  0.35070735  1.4259954 ]]
# [1]
# The person is diabetic

#===============================================================
