#!usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,\
    classification_report, confusion_matrix
# 完整的pipeline
'''preparing data'''
data = pd.read_csv('dtree_data.csv',header=None)
print(data.iloc[:,2].value_counts())
X = data.iloc[:,:-1].to_numpy()
y = data.iloc[:,-1].to_numpy()
# split data
X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.20, random_state=53)

'''training the model'''
clf = DecisionTreeClassifier(criterion='gini',random_state=53)
clf.fit(X_train,y_train)

'''making predictions'''
y_pred = clf.predict(X_test)
#print(y_pred)

'''evaluating the model'''
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

def visual(X,y):
    '''visualization'''
    # Plot the decision boundary. For that, we will asign a color
    # to each point in the mesh [x_min, x_max] x [y_min, y_max].
    x_min, x_max = X[:,0].min() - 0.1, X[:,0].max() + 0.1
    y_min, y_max = X[:,1].min() - 0.1, X[:,1].max() + 0.1
    h = 0.02 # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),\
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Put the result into a color plot
    plt.figure(2)
    plt.set_cmap(plt.cm.Paired)
    plt.pcolormesh(xx, yy, Z)
    # Plot the testing points
    for i in range(len(X)):
        if y[i] == 0:
            _ = plt.scatter(X[i,0], X[i,1], c='red', marker='+')
        else:
            _ = plt.scatter(X[i,0], X[i,1], c='blue', marker='x')
    plt.xlim()
    plt.ylim()
    plt.title('decision tree classifier model boundaries')

visual(X_test,y_test)
plt.show()
