import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from sklearn.pipeline import Pipeline
import matplotlib as mpl
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
#导入相关数据，这里导入鸢尾花数据集
iris = datasets.load_iris()
#查看数据集属性数据形状
chi_cun = iris['data'].shape
#查看数据集目标变量数据形状
target_chara = iris['target'].shape
#将数据集分割为训练集和测试集，这里分割比例是4:1。
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,test_size=0.2, random_state=0)
#查看训练集数据
train_data_chara = X_train.shape
#查看训练集数据
train_data_result = y_train.shape
#查看学习集数据
test__data_chara = X_test.shape
#查看学习集数据
test_data_result = y_test.shape

logisticregression = linear_model.LogisticRegression(penalty='l2',C=1.0)
model = logisticregression.fit(X_train,y_train)
result = logisticregression.predict(X_test)
fig = plt.figure(1)
#预测
idx_1 = np.where(result == 0)  # 找出第一类
p1 = plt.scatter(idx_1, result[idx_1], marker='.', color='black', s=10)
idx_2 = np.where(result == 1)  # 找出第二类
p2 = plt.scatter(idx_2, result[idx_2], marker='<', color='red', s=10)
idx_3 = np.where(result == 2)  # 找出第三类
p3 = plt.scatter(idx_3, result[idx_3], marker='4', color='green', s=20)
# ax1 = fig.add_subplot(121)
# p4 = ax1.scatter(iris.data[:,1],iris.data[:,2],marker='4', color='green', s=20)
#画图
plt.rcParams['font.sans-serif']=['SimHei']
plt.title("鸢尾花分类")
plt.xlabel("样本数")
plt.ylabel("分类")
plt.legend([p1, p2, p3], ['setosa', 'versicolor', 'vieginica'], bbox_to_anchor=(1, 1),loc=1)

## 特征与标签组合的散点可视化
## 不同的花萼长宽对应的不同花瓣长宽，散点分布。
iris_target = iris.target
iris_features = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_all = iris_features.copy() ##进行浅拷贝，防止对于原始数据的修改
iris_all['target'] = iris_target
sns.pairplot(data=iris_all,diag_kind='hist', hue= 'target')
x = X_train[:,[2,3]]
y = y_train
N, M = 200, 200     # 横纵各采样多少个值
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()   # 第0列的范围
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()   # 第1列的范围
t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, M)
x1, x2 = np.meshgrid(t1, t2)                    # 生成网格采样点
x_test = np.stack((x1.flat, x2.flat), axis=1)   # 测试点

lr = Pipeline([('sc', StandardScaler()),
                   ('poly', PolynomialFeatures(degree=10)),
                   ('clf', LogisticRegression()) ])
lr.fit(x, y.ravel())
y_hat = lr.predict(x)
y_hat_prob = lr.predict_proba(x)
np.set_printoptions(suppress=True)
mpl.rcParams['font.sans-serif'] = ['simHei']
mpl.rcParams['axes.unicode_minus'] = False
cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
y_hat = lr.predict(x_test)                  # 预测值
y_hat = y_hat.reshape(x1.shape)                 # 使之与输入的形状相同
plt.figure(facecolor='w')
plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)     # 预测值的显示
plt.scatter(x[:, 0], x[:, 1], c=y.flat, edgecolors='k', s=50, cmap=cm_dark)    # 样本的显示
plt.xlabel(u'花瓣长度', fontsize=14)
plt.ylabel(u'花瓣宽度', fontsize=14)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.grid()
patchs = [mpatches.Patch(color='#77E0A0', label='Iris-setosa'),mpatches.Patch(color='#FF8080', label='Iris-versicolor'),mpatches.Patch(color='#A0A0FF', label='Iris-virginica')]
plt.legend(handles=patchs, fancybox=True, framealpha=0.8)
plt.title(u'鸢尾花Logistic回归分类效果', fontsize=17)
plt.show()

