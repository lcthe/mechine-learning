"""
逻辑回归-鸢尾花数据集
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions

# "Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2

mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False

iris = load_iris()
# print(list(iris))

# sepal length,sepal width,petal length,petal width
X = iris.data[:, [2, 3]]
# X = iris.data
y = iris.target

# # 二分类
# y = iris.target != 2

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=1)

lr = LogisticRegression(penalty='l2', C=1.0)
lr.fit(train_X, train_y)
# 预测得到标签值
result = lr.predict(test_X)
# print(result)
print(lr.score(test_X, test_y))


# 原始数据图
plt.scatter(X[:, 0], X[:, 1], s=40, c=y)
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.title("逻辑回归-鸢尾花-原始数据")
plt.legend()
plt.show()

# 绘制预测值和真实值各自的标签
plt.plot(result, "ro", ms=10, label="预测值")
plt.plot(test_y, "go", label="真实值")
plt.title("逻辑回归-鸢尾花三分类")
plt.xlabel("样本序号")
plt.ylabel("类别")
plt.legend()
plt.show()


plt.figure()
fig = plot_decision_regions(X=test_X, y=test_y, clf=lr, legend=2)
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.title("逻辑回归-鸢尾花-决策边界")
handles, labels = fig.get_legend_handles_labels()
fig.legend(handles,
           ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
           framealpha=0.3, scatterpoints=1)
plt.show()


