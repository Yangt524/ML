# by : Yangt
# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


# np.random.seed(123)
# 加载iris数据集
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]    # 为了绘图方便，此处只选取前两个特征进行训练
Y = iris.target

# 生成三个学习器
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=8)
clf3 = SVC(kernel='rbf', probability=True)

eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=[3, 2, 1])
# print(eclf._estimator_type)
clf1.fit(X, Y)
clf2.fit(X, Y)
clf3.fit(X, Y)

eclf.fit(X, Y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# 生成坐标矩阵
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# 画图
f, axarr = plt.subplots(2, 2, sharex='col', sharey='row')
# product():运算笛卡尔积，返回值为元组形式
for idx, clf, tt in zip(product([0, 1], [0, 1]), [clf1, clf2, clf3, eclf], ['DT', 'KNN', 'SVM', 'Soft Voting']):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.5)    # contourf():画等高线，xx和yy为网格矩阵；Z为高度，及Z=f(x,y)；alpha参数表示颜色的深浅
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=Y, s=20, edgecolor='k')
    axarr[idx[0], idx[1]].set_title(tt)

plt.show()
