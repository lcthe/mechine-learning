import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import datasets
from IPython.display import display

mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False
data = pd.read_csv(r"boston.csv")
# data.head()
class LinearRegression:
    def __init__(self, alpha, times):
        """

        :param alpha: float
            学习率 控制步长
        :param times:int
            迭代次数
        """
        self.alpha = alpha
        self.times = times

    def fit(self, X, y):
        """
        根据数据开始训练
        :param X: 类数组类型，形状：[样本数量， 特征数量]
            待训练的样本特征数量
        :param y: 类数组类型 形状：[样本数量]
            目标值（标签信息）
        """
        X = np.asarray(X)
        y = np.asarray(y)
        # 创建权重的向量，初始值为0,长度比特征向量多1（截距）
        self.w_ = np.zeros(1 + X.shape[1])
        # 创建损失列表，用来保存每次迭代后的损失值，损失值计算 （预测值 - 真实值）的平方和 除以 2
        self.loss_ = []

        # 进行循环，多次迭代，在每次迭代中调整权重值，使得损失值不断减小
        for i in range(self.times):
            # 计算预测值
            y_hat = np.dot(X, self.w_[1:]) + self.w_[0]
            # 计算真实值与预测值之间的差距
            error = y - y_hat
            # 计算损失值
            self.loss_.append(np.sum(error**2)/2)
            # 根据差距error调整权重w_，根据公式，调整为 权重[j] = 权重[j] + 学习率 * sum((y - y_hat) * x(j))
            self.w_[0] += self.alpha * np.sum(error * 1)
            self.w_[1:] += self.alpha * np.dot(X.T, error)

    def predict(self, X):
        """
        根据参数传递的样本，进行预测
        :param X: 类数组类型，形状：[样本数量， 特征数量]
            待测试的样本
        :return result: 数组类型
            预测的结果
        """
        X = np.asarray(X)
        result = np.dot(X, self.w_[1:]) + self.w_[0]
        return result

# 未标准化处理的数据
# lr = LinearRegression(alpha=0.001, times=20)
# t = data.sample(len(data), random_state=0)
# train_X = t.iloc[:400, :-1]
# train_y = t.iloc[:400, -1]
# test_X = t.iloc[400:, :-1]
# test_y = t.iloc[400:, -1]
#
# lr.fit(train_X, train_y)
# result = lr.predict(test_X)

#
# # display(np.mean((result - test_y) ** 2))
# # display(lr.w_)
# # display(lr.loss_)

class StandardScaler:
    """
    该类对数据进行标准化处理
    """
    def fit(self, X):
        """
        根据传入的样本，计算每一个特征列的均值和标准差
        :param X: 类数组类型
            训练数据，用来计算均值和标标准差
        """
        X = np.asarray(X)
        self.std_ = np.std(X, axis=0)
        self.mean_ = np.mean(X, axis=0)

    def transform(self,X):
         """
         对给定的数据X，进行标准化处理
         :param X: 类数组类型
            待转换的数据
         :return:类数组类型
            参数X转换成标准正态分布后的结果
         """
         return(X-self.mean_) / self.std_

    def fit_transform(self,X):
         """
         对数据进行训练，并转换，返回转换之后的结果
         :param X:类数组类型
            待转换的数据
         :return result:；类数组类型
            参数X转换成标准正态分布后的结果
         """
         self.fit(X)
         return self.transform(X)

# 考虑每个特征的数据量级不同，从而对每个特征标准化处理
# alpha怎么选取？
lr = LinearRegression(alpha=0.0005, times=20)
t = data.sample(len(data), random_state=0)
train_X = t.iloc[:400, :-1]
train_y = t.iloc[:400, -1]
test_X = t.iloc[400:, :-1]
test_y = t.iloc[400:, -1]

s = StandardScaler()
train_X = s.fit_transform(train_X)
test_X = s.transform(test_X)

s2 = StandardScaler()
train_y = s2.fit_transform(train_y)
test_y = s2.transform(test_y)

lr.fit(train_X, train_y)
result = lr.predict(test_X)

display(np.mean((result - test_y) ** 2))
display(lr.w_)
display(lr.loss_)

# 数据可视化

plt.figure(figsize=(10,10))
plt.plot(result, "ro-", label="预测值")
plt.plot(test_y.values, "go-", label="预测值") # pandas读取时serise类型，我们需要转为ndarray
plt.title("线性回归预测-梯度下降")
plt.xlabel("测试机样本序号")
plt.ylabel("预测房价")
plt.legend()
plt.show()
# plt.figure(1)


plt.plot(range(1,lr.times+1),lr.loss_, "o-")
plt.show()
# plt.figure(2)
#
# # 因为房价更新涉及多个维度，不方便可视化
# # 为了可视化，我们只选择其中一个维度（RM），并画出直线，进行拟合
# lr = LinearRegression(alpha=0.0005, times=20)
# t = data.sample(len(data), random_state=0)
# train_X = t.iloc[:400, 6:7]
# train_y = t.iloc[:400, -1]
# test_X = t.iloc[400:, 5:6]
# test_y = t.iloc[400:, -1]
# # 标准化
# ss = StandardScaler()
# train_X = ss.fit_transform(train_X)
# test_X = ss.fit_transform(test_X)
# ss2 = StandardScaler()
# train_y = ss2.fit_transform(train_y)
# test_y = ss2.fit_transform(test_y)
# lr.fit(train_X, train_y)
# result = lr.predict(test_X)
# # display(result)
# display(np.mean((result - test_y)**2))
#
# # 展示rm对对于价格的影响
# plt.scatter(train_X["rm"], train_y)
# # 展示权重
# display(lr.w_)
# # 构建方程 y = w0 + x*w1 = -3.05755421e-16 + x *  6.54993957e-01
# x = np.arange(-5,5,0.1)
# y = -3.05755421e-16 + x *  6.54993957e-01
# plt.plot(x, y, "g") # 绿色直线显示我们的拟合直线
# # *********  x.reshape(-1,1) 把一维转为二位 ****************
# plt.plot(x, lr.predict(x.reshape(-1,1)), 'r')