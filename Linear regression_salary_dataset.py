# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:39:23 2020

@author: SrinivaS
"""

#conda install sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Salary_Data.csv')

#separating into dependent and independent feature
x = data.iloc[:, :-1].values  #.values is used, so the datatype is also assigned properly
y = data.iloc[:, 1:].values   

#splitting the main data set for training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0) #you can give either test size or train size

#Fitting simple linear regression to the training data set : Model creation
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)       #we are using the train dataset to create our model.

y_predict = lr.predict(x_test) #after training we are predicting the test data set

y_predict_b = lr.predict([[11]])  #While predicting the value which is not in the array then you need to provide into the array like y_predict_val = simplelinearRegression.predict([[11.0]]) instead of just (11)

#visulaizing the training data set
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, lr.predict(x_train), color='blue')
#plt.show()

#visualising the test data set
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, lr.predict(x_train), color='blue')
#plt.plot(x_test, lr.predict(x_test), color='blue')
