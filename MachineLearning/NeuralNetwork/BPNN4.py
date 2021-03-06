# by : Yangt
# -*- coding: utf-8 -*-

import time

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

np.random.seed(0)


def sigmoid(x):
    if x > 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


def dsigmoid(x):
    return x * (1 - x)


def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))


def dtanh(x):
    return 1 - (x ** 2)


class BPNeuralNetwork:
    def __init__(self, ni, nh, no):
        self.input_n = ni + 1
        self.hidden_n = nh + 1
        self.output_n = no
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n
        self.input_hidden_weights = np.random.randn(self.input_n, self.hidden_n) / np.sqrt(self.input_n)
        self.hidden_output_weights = np.random.randn(self.hidden_n, self.output_n) / np.sqrt(self.hidden_n)

    def predict(self, data):
        for i in range(self.input_n - 1):
            self.input_cells[i] = data[i]

        for h in range(self.hidden_n - 1):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_hidden_weights[i][h] * self.input_cells[i]
            self.hidden_cells[h] = sigmoid(total)

        for o in range(self.output_n):
            total = 0.0
            for h in range(self.hidden_n):
                total += self.hidden_output_weights[h][o] * self.hidden_cells[h]
            self.output_cells[o] = sigmoid(total)

        return self.output_cells

    def back_propagation(self, data, target, learning_rate):
        self.predict(data)

        g = [0.0] * self.output_n
        for o in range(self.output_n):
            g[o] = (target[o] - self.output_cells[o]) * dsigmoid(self.output_cells[o])

        e = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            total = 0.0
            for o in range(self.output_n):
                total += self.hidden_output_weights[h][o] * g[o]
            e[h] = total * dsigmoid(self.hidden_cells[h])

        # ???????????????????????????????????????
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_hidden_weights[i][h] += learning_rate * e[h] * self.input_cells[i]

        # ???????????????????????????????????????
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.hidden_output_weights[h][o] += learning_rate * g[o] * self.hidden_cells[h]

        # ????????????
        error = 0.0
        for o in range(len(target)):
            error += 0.5 * (target[o] - self.output_cells[o]) ** 2
        return error

    def train(self, dataSet, targets, data_test, target_test, limit, learning_rate=0.1):
        train_loss = []
        test_loss = []
        acc = []
        for i in range(limit):
            error = 0.0
            for data, target in zip(dataSet, targets):
                error += self.back_propagation(data, target, learning_rate)
            train_loss.append(error/len(targets))

            test_loss.append(self.test(data_test, target_test))
        x = np.linspace(1, limit, limit)
        plt.plot(x, train_loss, 'r', label=u'train loss')
        plt.plot(x, test_loss, 'b', label=u'test loss')
        # plt.axis([-1, limit, 0, 0.05])
        plt.legend(loc='best')
        plt.show()

    def test(self, datas, targets):
        count = 0
        # target_pred = []
        # error_data = []
        error_test = 0.0
        for i in range(len(datas)):
            predict_y = self.predict(datas[i])
            # target_pred.append(predict_y)
            error = 0.0
            for o in range(len(targets[i])):
                error += 0.5 * (targets[i][o] - predict_y[o]) ** 2
            # print('y:{}==>predict_y:{}\terror:{}'.format(targets[i], predict_y, error))
            if error < 0.05:
                count += 1
            error_test += error / len(datas)

        return error_test
        # print('???????????????????????????', error_test)
        # print('accuracy:%.2f%%' % (100 * count/len(targets)))


if __name__ == '__main__':
    nn = BPNeuralNetwork(30, 6, 1)
    # ??????????????????
    breast_cancer = datasets.load_breast_cancer()
    dataSet = breast_cancer.data.tolist()
    targets = breast_cancer.target
    targets = targets.reshape(len(targets), 1).tolist()

    dataSet = preprocessing.scale(dataSet)

    # nn = BPNeuralNetwork(4, 7, 3)
    # iris = datasets.load_iris()
    # dataSet = iris.data.tolist()
    # ts = iris.target
    # targets = []
    # for i in range(len(ts)):
    #     if ts[i] == 0:
    #         targets.append([0, 0, 1])
    #     elif ts[i] == 1:
    #         targets.append([0, 1, 0])
    #     else:
    #         targets.append([1, 0, 0])
    # # dataSet = preprocessing.scale(dataSet)
    # ?????????????????????
    temp = list(zip(dataSet, targets))
    np.random.shuffle(temp)
    dataSet, targets = zip(*temp)

    data_train, data_test, target_train, target_test = train_test_split(dataSet, targets,
                                                                        test_size=0.3, random_state=456)
    time_start = time.time()
    nn.train(data_train, target_train, data_test, target_test, 1000, 0.1)
    time_end = time.time()

    print('????????????:', time_end - time_start, 's')

    # time_start = time.time()
    # nn.test(data_test, target_test)
    # time_end = time.time()
    #
    # print('???????????????', time_end - time_start, 's')
    # nn.train(dataSet, targets, 1000, 0.1)
    # nn.test(dataSet, targets)
