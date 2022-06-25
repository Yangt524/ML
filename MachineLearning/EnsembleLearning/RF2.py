# by : Yangt
# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn.datasets import make_hastie_10_2
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


np.random.seed(123)
# # 加载iris数据集
# breast_cancer = datasets.load_breast_cancer()
# X = breast_cancer.data    # 为了绘图方便，此处只选取前两个特征进行训练
# Y = breast_cancer.target
# # 随机打乱数据集
# temp = list(zip(X, Y))
# np.random.shuffle(temp)
# X, Y = zip(*temp)
X, Y = make_hastie_10_2(n_samples=30000, random_state=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=123)
X_train, X_test, Y_train, Y_test = np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)

clf = DecisionTreeClassifier(max_depth=4)
eclf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train, Y_train)
eclf.fit(X_train, Y_train)

# x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
# y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
# # 生成坐标矩阵
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
#
# f, axarr = plt.subplots(2, sharex='col', sharey='row', figsize=(8, 6))
# # product():运算笛卡尔积，返回值为元组形式
# for idx, clf, tt in zip([0, 1], [clf, eclf], ['DT', 'RF']):
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#
#     axarr[idx].contourf(xx, yy, Z, alpha=0.5)    # contourf():画等高线，xx和yy为网格矩阵；Z为高度，及Z=f(x,y)；alpha参数表示颜色的深浅
#     axarr[idx].scatter(X_train[:, 0], X_train[:, 1], c=Y_train, s=20, edgecolor='k')
#     axarr[idx].set_title(tt)


res = clf.predict(X_test)
report = classification_report(Y_test, res)

e_res = eclf.predict(X_test)
# print(res)
e_report = classification_report(Y_test, e_res)
print('DT：\n', report)
print('RF: \n', e_report)


# plt.show()




