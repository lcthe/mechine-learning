import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

"""
朴素贝叶斯分类器-西瓜数据集
"""

def createDataSet():
    """
    色泽 1-3代表 浅白 青绿 乌黑
    根蒂 1-3代表 稍蜷 蜷缩 硬挺
    敲声 1-3代表 清脆 浊响 沉闷
    纹理 1-3代表 清晰 稍糊 模糊
    脐部 1-3代表 平坦 稍凹 凹陷
    触感 1-2代表 硬滑 软粘
    好瓜 1 代表 是 0 代表 不是
    """
    dataSet = [
        # 好瓜
        [2, 2, 2, 1, 3, 1, 0.697, 0.460, 1],
        [3, 2, 3, 1, 3, 1, 0.744, 0.376, 1],
        [3, 2, 2, 1, 3, 1, 0.634, 0.264, 1],
        [2, 2, 3, 1, 3, 1, 0.608, 0.318, 1],
        [1, 2, 2, 1, 3, 1, 0.556, 0.215, 1],
        [2, 1, 2, 1, 2, 2, 0.403, 0.237, 1],
        [3, 1, 2, 2, 2, 2, 0.481, 0.149, 1],
        [3, 1, 2, 1, 2, 1, 0.437, 0.211, 1],
        # 坏瓜
        [3, 1, 3, 2, 2, 1, 0.666, 0.091, 0],
        [2, 3, 1, 1, 1, 2, 0.243, 0.267, 0],
        [1, 3, 1, 3, 1, 1, 0.245, 0.057, 0],
        [1, 2, 2, 3, 1, 2, 0.343, 0.099, 0],
        [2, 1, 2, 2, 3, 1, 0.639, 0.161, 0],
        [1, 1, 3, 2, 3, 1, 0.657, 0.198, 0],
        [3, 1, 2, 1, 2, 2, 0.360, 0.370, 0],
        [1, 2, 2, 3, 1, 1, 0.593, 0.042, 0],
        [2, 2, 3, 2, 2, 1, 0.719, 0.103, 0]
    ]

    # 特征值列表
    labels = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '密度', '含糖率', '好瓜']

    return dataSet, labels

dataSets, labels = createDataSet()
# print(dataSets)
X = np.asarray(dataSets)
y = np.asarray(dataSets)
X = X[:, :-1]
y = y[:, -1]
# print(y)
# 训练朴素贝叶斯分类器（高斯函数）模型
bayes = GaussianNB()
bayes.fit(X, y)

# 读取测试数据
test = np.asarray([2, 2, 2, 1, 3, 1, 0.597, 0.160])
# 转二维
test = test.reshape(1, -1)

# 利用模型预测
result = bayes.predict(test)
# 输出结果
if result == 1:
    print("它是个好瓜")
else:
    print("它是个坏瓜")

