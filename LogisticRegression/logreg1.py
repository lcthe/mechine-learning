#!usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,\
    classification_report, confusion_matrix

# 完整的pipeline
'''preparing data'''
data = pd.read_csv('nbayes_data.csv',header=None)
print(data.iloc[:,2].value_counts())
X = data.iloc[:,:-1].to_numpy()
y = data.iloc[:,-1].to_numpy()
# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=53)

'''visualization'''
def visual(X,y):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    h = 0.02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    # Plot the training points
    for i in range(len(X)):
        if y[i] == 0:
            _ = plt.scatter(X[i,0], X[i,1], c='red', marker='x')
        elif y[i] == 1:
            _ = plt.scatter(X[i,0], X[i,1], c='blue', marker='+')
        elif y[i] == 2:
            _ = plt.scatter(X[i,0], X[i,1], c='green', marker='o')
        else:
            _ = plt.scatter(X[i,0], X[i,1], c='c', marker='^')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

'''training the model'''
clf = LogisticRegression(C=1)
clf.fit(X_train,y_train)
plt.figure(1, figsize=(10, 7))
visual(X_test,y_test)
plt.title('logistic regression classifier model boundariers(C=1)')

'''making predictions'''
y_pred = clf.predict(X_test)

'''evaluating the model'''
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

'''overfiting'''
clf = LogisticRegression(C=1000)
clf.fit(X_train,y_train)
plt.figure(2, figsize=(10, 7))
visual(X_test,y_test)
plt.title('logistic regression classifier model boundariers(C=1000)')
plt.show()




