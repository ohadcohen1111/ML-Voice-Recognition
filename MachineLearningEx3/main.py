# This is a sample Python script.


# evaluate adaboost algorithm for classification


import seaborn as sns
import pandas as pd
import numpy as np



from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn import datasets, metrics
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier



#Load data
# iris = datasets.load_iris("/Desktop/Studies/Machine-Learning/Ex3/HC_Body_Temperature.txt")
# X = iris.data
# y = iris.target
# # define dataset
# # X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=6)
# # define the model
# model = AdaBoostClassifier()
# # evaluate the model
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# # report performance
# print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
#
from Point import Point
from sklearn.model_selection import train_test_split



# p = Point(0,1)
# print(p)

# pointList = []
# f = open("HC_Body_Temperature.txt", "r")
# k = f.readlines()
# for i in k:
#     j = i.split()
#     # print(j[0])
#     pointList.append(Point(j[0],j[2],j[1]))
#
# for j in pointList:
#     print(j)
#
# train, test = train_test_split(pointList, test_size=0.5)
# # print("Train size: " + len(train) + "Test size: " + len(test))
# print(len(train))
# print(len(test))
#
# abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)
#
# # Train Adaboost Classifer
# model = abc.fit(train, test)
# y_pred = model.predict(test)
# print("Accuracy:",metrics.accuracy_score(train, y_pred))
#
#
#

# data = datasets.load_iris()
# print(data)
