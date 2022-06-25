"""
@author:Yangt
@file:DBSCAN.py
@time:2021/11/22
@version:1.0
"""
import numpy as np
from sklearn.datasets import make_blobs
import queue
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

np.random.seed(0)


def distance(vectorA, vectorB):
    """
    计算两点之间的欧氏距离
    :param vectorA:
    :param vectorB:
    :return: 距离
    """
    vectorA, vectorB = np.array(vectorA), np.array(vectorB)
    temp = vectorA - vectorB
    return np.sqrt((temp**2).sum())


def load_data():
    """
    加载数据
    Returns:

    """
    X, Y = make_circles(n_samples=2000, shuffle=True, noise=0.1, factor=0.01)
    X = [list(x) for x in X]
    Y = list(Y)
    return X, Y


def get_diff(A, B):
    if type(A) != list:
        A = A.tolist()
    for b in B:
        if b in A:
            A.remove(b)
    return A


def get_intersect(A, B):
    res = []
    for a in A:
        if a in B:
            res.append(a)
    return res


class DBSCAN:
    def __init__(self, epsilon, MinPts):
        self.epsilon = epsilon
        self.MinPts = MinPts
        self.omega = None
        self.label = None
        self.C = None

    def fit(self, X):
        n = len(X)
        omega = []
        for i in range(n):
            N = [X[i]]
            for j in range(n):
                d = distance(X[i], X[j])
                if d <= self.epsilon:
                    N.append(X[j])
            if len(N) >= self.MinPts:
                omega.append(X[i])
        self.omega = omega[:]
        C = self.predict(X)
        self.C = C
        # print(C)

    def predict(self, X):
        k = 0
        Gamma = X[:]
        omega = self.omega[:]
        C = []
        while len(omega) != 0:
            Gamma_old = Gamma[:]
            index = np.random.choice(len(omega))
            o = omega[index]
            Q = queue.Queue()
            Q.put(o)

            Gamma = get_diff(Gamma, [o])        # 差集函数
            while not Q.empty():
                q = Q.get()
                N = [q]
                for j in range(len(Gamma_old)):
                    d = distance(q, Gamma_old[j])
                    if d <= self.epsilon:
                        N.append(Gamma_old[j])
                if len(N) >= self.MinPts:
                    deltas = get_intersect(Gamma, N)       # 交集函数
                    for delta in deltas:
                        Q.put(delta)
                    Gamma = get_diff(Gamma, deltas)
            k = k+1
            C.append(get_diff(Gamma_old, Gamma))
            omega = get_diff(omega, C[k-1])
        print(k)
        return C


if __name__ == '__main__':
    X, Y =load_data()
    model = DBSCAN(0.08, 10)
    model.fit(X)
    f, axarr = plt.subplots(2, 1)
    noise = X[:]
    for c in model.C:
        noise = get_diff(noise, c)
        c = np.array(c)
        axarr[0].scatter(c[:, 0], c[:, 1], edgecolor='k', s=20)

    if len(noise) != 0:
        noise = np.array(noise)
        axarr[0].scatter(noise[:, 0], noise[:, 1], c='k', marker='^')
    # omega = model.omega[:]
    # omega = np.array(omega)
    # axarr[0].scatter(omega[:, 0], omega[:, 1], c='r', marker='^')

    X, Y = np.array(X), np.array(Y)
    axarr[1].scatter(X[:, 0], X[:, 1], c=Y, s=20, edgecolor='k')
    plt.show()


