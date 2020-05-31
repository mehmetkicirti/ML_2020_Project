# -*- coding: utf-8 -*-
"""
Created on Sat May 28 14:35:23 2020

@author: mehmet
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import sklearn.linear_model as linearModels
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn import metrics

names=['x1','x2','x3','x4','x5','x6','Y']
dataSheet = pd.read_csv("CE_475_DataSheet.csv", names=names,nrows=101)
willPredict = pd.read_csv("CE_475_DataSheet.csv",skiprows=range(1,101),nrows=20)
X = dataSheet.iloc[1:,0:6].values
Y = dataSheet.iloc[1:,6].values

find_X= willPredict.iloc[0:,1:7].values

def highestAccuracyOrderSorted(X,Y):
    # Which column is a best performance for these datasheet 
    # use lasso regression
    # all features such as until elimination to last one
    l_reg = linearModels.Lasso(alpha=0.01)
    rfe = RFE(l_reg,n_features_to_select=1)
    rfe.fit(X,Y)
    #ranking_ feature gives us which column is a best accuracy
    data = dict(zip(['x1','x2','x3','x4','x5','x6'],rfe.ranking_))
    return sorted([(round(value,2),key) for (key,value) in data.items()])

#SORTED ACCURACY 
print('Which column give best accuracy sorted => ',highestAccuracyOrderSorted(X,Y))   

#Configure dataSheet Again According to above Result => [x1,x3,x2,x6] I will take
x2 = dataSheet.iloc[1:,1].values
x3 = dataSheet.iloc[1:,2].values
x6 = dataSheet.iloc[1:,5].values
x1 = dataSheet.iloc[1:,0].values

# In Here prepared Dataset
X_New = np.array([x1,x2,x3,x6]).T
# Now prepare our test and train data and getting test data size as 20 
#Each time divided same data random state
X_train,X_test,Y_train,Y_test = train_test_split(X_New,Y,test_size=0.2,shuffle=True,random_state=0)

def linearRegression(train_x,train_y,test_x,test_y):
    model = linearModels.LinearRegression()
    #Linear Regression putting with our train data's
    l_reg = model.fit(train_x,train_y)
    #Predicted with our test data
    predict = model.predict(test_x)
    #Cross Validation Score found better predicted which testing error  
    cvScore = crossValidScore(X_New,Y, l_reg)
    
    getResults(test_y,predict,"Linear Regression",cvScore)   

def decisionTree(train_x,train_y,test_x,test_y):
    dt = DecisionTreeRegressor(random_state=0,max_depth=2)
    # fitted data
    dt_reg = dt.fit(train_x,train_y)
    predict = dt.predict(test_x)
    cvScore = crossValidScore(X_New, Y, dt_reg) 
    
    getResults(test_y,predict,"Decision Tree",cvScore)
    
def lassoRegression(train_x,train_y,test_x,test_y):
    model = linearModels.Lasso(alpha=0.01)
    #Lasso Regression putting with our train data's
    l_reg = model.fit(train_x,train_y)
    #Predicted with our test data
    predict = model.predict(test_x)
    #Cross Validation Score found better predicted which testing error  
    cvScore = crossValidScore(X_New,Y, l_reg)
    
    getResults(test_y,predict,'Lasso Regression',cvScore)

def randomForestRegression(train_x,train_y,test_x,test_y):
    #n_estimators => How many get tree
    model = RandomForestRegressor(n_estimators=40,max_features='auto')
    rf_reg = model.fit(train_x, train_y)
    predict = model.predict(test_x)
    cvScore = crossValidScore(X_New, Y, rf_reg)
    getResults(test_y, predict, "Random Forest Regression",cvScore)
    predicted_y = findUndefinedValues(rf_reg,test_x)
    print("\nPredicted Y Last 20 => \n",predicted_y)
    exportToCsvFile(predicted_y)

def polynomialRegression(train_x,train_y,test_x,test_y):
    model = linearModels.LinearRegression()
    # Fitting Polynomial Regression to the dataset 
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(train_x)
    poly_reg = model.fit(X_poly, train_y)
    #Predicted data
    predict = model.predict(poly.fit_transform(test_x))
    #Cross Score 
    cvScore = crossValidScore(X_New, Y, poly_reg)
    
    getResults(test_y, predict,  "Polynomial Regression",cvScore)
    
def crossValidScore(X,Y,regFitData):
    cvScore = cross_val_score(regFitData,X,Y,cv=9)
    return cvScore.mean()

def getResults(test_y,predict,title,*vartuple):
    print('\n------------------ {0} -------------------'.format(title))
    # Mean Absolute Error
    print("Mean Absolute  Error {0} : {1} ".format(title, metrics.mean_absolute_error(test_y, predict)))
    
    # Mean Squared Error
    print('MSE for {0} : {1}'.format(title,metrics.mean_squared_error(test_y,predict)))
    # Root Mean Squared Error
    print('Root MSE for {0} : {1}'.format(title,np.sqrt(metrics.mean_squared_error(test_y,predict))))
    # R Squared Score
    print('R2 Score for {0} : {1}'.format(title,metrics.r2_score(test_y,predict)))
    for var in vartuple:
            # Cross Validation Score Accuracy
        print('CVScore for {0} : {1}'.format(title,var.mean()))

# Will be predict last 20 values 
def findUndefinedValues(estimator,value):
    yPredictedValues = estimator.predict(value)
    return yPredictedValues
# Export as excel file into project folder
def exportToCsvFile(predicted_y):
    values = np.column_stack((find_X,predicted_y))
    df = pd.DataFrame(values,columns=names)
    return df.to_excel('predictedValues.xlsx',index=False,header=True)

def supportVectorMachine(train_x,train_y,test_x,test_y):
    model = SVC(kernel='linear')
    model.fit(train_x,train_y)
    predict = model.predict(test_x)
    getResults(test_y, predict, "Support Vector Machine")
    
linearRegression(X_train, Y_train, X_test,Y_test)    
lassoRegression(X_train, Y_train, X_test,Y_test)
randomForestRegression(X_train, Y_train, X_test,Y_test) 
polynomialRegression(X_train, Y_train, X_test,Y_test)
supportVectorMachine(X_train, Y_train, X_test,Y_test)
decisionTree(X_train, Y_train, X_test,Y_test)    

   