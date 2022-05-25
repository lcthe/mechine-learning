import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False
"""
    选择鸢尾花数据集
"""
data = pd.read_csv(r"iris.arff.csv")
# print(data)

# 删除重复的列
data.drop_duplicates(inplace=True)


# 三个class Iris-setosa,Iris-versicolor,Iris-virginica
data["class"] = data["class"].map({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})

# 只选取类型0和类型1进行逻辑回归的二分类
data = data[data["class"] != 2]
# print(len(data))

class LogisticRegression:
    """
    逻辑回归算法
    """
    def __init__(self, alpha, times):
        """
        初始化方法
        :param alpha:float
            学习率，更新步长
        :param times: int
            迭代次数
        """
        self.alpha = alpha
        self.times = times

    def sigmoid(self, z):
        """
        sigmoid函数的实现
        :param z: float
            自变量，值为z = w.T * x
        :return:float
            p值为[0，1]之间，用来作结果的预测
            当 s>=0,5(z>=0) 判定为类型1，否则判定为0
        """
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        """
        根据提供的训练数据，进行训练
        :param X: 类数组类型，形状:[样本数量，特征数量]
            待训练的样本特征属性
        :param y: 类数组类型，形状：[样本数量]
            每个样本的目标值（标签）
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # 创建权重向量，初始值为0,长度比特征数多1（截距）
        self.w_ = np.zeros(1 + X.shape[1])

        # 创建损失列表，保存每次叠迭代后的损失值
        self.loss_ = []

        for i in range(self.times):
            z = np.dot(X,self.w_[1:])+self.w_[0]
            # 计算概率值
            p = self.sigmoid(z)
            # 根据逻辑回归的代价函数（目标函数）计算损失值
            # 逻辑回归的代价函数（目标函数）：
            # J(w)=-sum(yi * log(sigmoid(z))-(1-yi)*log(1-sigmoid(zi))) i从1到n
            cost = -np.sum(y * np.log(p) + (1 - y) * np.log(1-p))
            self.loss_.append(cost)

            # 调整权重值, 根据公式调整为: 权重(j列) = 权重(j列) + 学习率 * sum((y-s(z))*x(j))
            self.w_[0] += self.alpha * np.sum(y - p)
            self.w_[1:] += self.alpha * np.dot(X.T, y-p)

    def predict_proba(self, X):
        """
        根据参数传递的样本，对样本数据进行预测
        :param X: 类数组类型 形状：[样本数量，特征数量]
            待测试的样本特征
        :return: 数组类型
            预测的结果（概率值）
        """
        X = np.asarray(X)
        z = np.dot(X, self.w_[1:]) + self.w_[0]
        p = self.sigmoid(z)
        # 待预测结果变为二维，便于后续的拼接
        p = p.reshape(-1, 1)
        # 将两个数组拼接，横向拼接
        return np.concatenate([1-p, p], axis=1)

    def predict(self,X):
        """
        根据参数传递的样本，对样本数据进行预测
        :param X: 类数组类型 形状：[样本数量，特征数量]
            待测试的样本特征
        :return: 数组类型
            预测的结果（标签）
        """
        return np.argmax(self.predict_proba(X), axis=1)

# if __name__ == "__main":

t1 = data[data["class"] == 0]
t2 = data[data["class"] == 1]
t1 = t1.sample(len(t1), random_state=0)
t2 = t2.sample(len(t2), random_state=0)
train_X = pd.concat([t1.iloc[:20, :-1], t2.iloc[:20, :-1]], axis=0)
train_y = pd.concat([t1.iloc[:20, -1], t2.iloc[:20, -1]], axis=0)
test_X = pd.concat([t1.iloc[:20, :-1], t2.iloc[:20, :-1]], axis=0)
test_y = pd.concat([t1.iloc[:20, -1], t2.iloc[:20, -1]], axis=0)

# 数据是同一个数量级，不用标准化处理
lr = LogisticRegression(alpha=0.01, times=20)
lr.fit(train_X, train_y)
# 预测的概率值
# print(lr.predict_proba(test_X))

# 预测的标签值
# print(lr.predict(test_X))

result = lr.predict(test_X)

#  计算准确率
# print(np.sum(result == test_y) / len(test_y))


# 绘制预测值
plt.plot(result, "ro", ms=10, label="预测值")

# 绘制真实值
plt.plot(test_y.values, "go", label="真实值")
plt.title("逻辑回归")
plt.xlabel("样本序号")
plt.ylabel("类别")
plt.legend()
plt.show()
plt.figure(1)

# 绘制损失值
plt.plot(range(1, lr.times+1), lr.loss_, "go-")
plt.title("损失值")
plt.xlabel("样本序号")
plt.ylabel("损失值")
plt.legend()
plt.show()
plt.figure(2)