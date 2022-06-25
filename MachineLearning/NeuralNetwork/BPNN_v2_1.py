# by : Yangt
# -*- coding: utf-8 -*-
import numpy as np
import time

from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing

np.random.seed(0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def partial_sigmoid(x):
    return x * (1 - x)


class BPNeuralNetwork:
    def __init__(self, ni, nh, no):
        self.input_n = ni + 1
        self.hidden_n = nh + 1
        self.output_n = no
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        self.input_hidden_weights = np.random.randn(self.input_n, self.hidden_n) / np.sqrt(self.input_n)
        self.hidden_output_weights = np.random.randn(self.hidden_n, self.output_n) / np.sqrt(self.hidden_n)
        self.input_hidden_weights_correction = np.zeros((2, self.input_n, self.hidden_n), dtype=float)
        self.hidden_output_weights_correction = np.zeros((2, self.hidden_n, self.output_n), dtype=float)

    def predict(self, datas):
        # 根据len(datas)的大小调整cells的行数
        self.input_cells = np.ones((len(datas), self.input_n), dtype=float)
        self.hidden_cells = np.ones((len(datas), self.hidden_n), dtype=float)
        self.output_cells = np.ones((len(datas), self.output_n), dtype=float)
        
        # 激活输入层
        # bias_cells = np.ones(len(datas), dtype=float)
        self.input_cells[:, :-1] = datas

        # 激活隐含层
        self.hidden_cells[:, :-1] = sigmoid(np.dot(self.input_cells, self.input_hidden_weights[:, :-1]))

        # 激活输出层
        self.output_cells = sigmoid(np.dot(self.hidden_cells, self.hidden_output_weights))

    def get_loss(self, targets):
        loss = np.dot((targets - self.output_cells) ** 2, 0.5 * np.ones((self.output_n, 1)))

        return loss

    def get_acc(self, loss, targets):
        count = 0
        print('预测出错的数据：')
        for i in range(len(loss)):
            if loss[i] < 0.05:
                count += 1
            else:
                print(targets[i], '====>', self.output_cells[i])
        acc = float(count) / len(loss)
        return acc

    def back_propagation(self, targets, learning_rate, correct_rate=0.0):
        # 计算隐层到输出层之间的残差
        hidden_output_delta = (targets - self.output_cells) * partial_sigmoid(self.output_cells)

        # 计算输入层到隐含层之间的残差
        input_hidden_delta = (np.dot(hidden_output_delta, self.hidden_output_weights.T)) * partial_sigmoid(self.hidden_cells)

        # 更新隐层与输出层之间的权值
        change = learning_rate * np.dot(self.hidden_cells.T, hidden_output_delta)
        self.hidden_output_weights += change + correct_rate * self.hidden_output_weights_correction[1]
        self.hidden_output_weights_correction[0] = self.hidden_output_weights_correction[1]
        self.hidden_output_weights_correction[1] = change

        # 更新输入层与隐层之间的权值
        change = learning_rate * np.dot(self.input_cells.T, input_hidden_delta)
        self.input_hidden_weights += change + correct_rate * self.input_hidden_weights_correction[1]
        self.input_hidden_weights_correction[0] = self.input_hidden_weights_correction[1]
        self.input_hidden_weights_correction[1] = change

    def back_one(self):
        self.hidden_output_weights -= self.hidden_output_weights_correction[1]
        self.input_hidden_weights -= self.input_hidden_weights_correction[1]
        self.hidden_output_weights_correction[1] = self.hidden_output_weights_correction[0]
        # self.hidden_output_weights_correction[0] = np.zeros((self.hidden_n, self.output_n), dtype=float)
        self.input_hidden_weights_correction[1] = self.input_hidden_weights_correction[0]
        # self.input_hidden_weights_correction[0] = np.zeros((self.input_n, self.hidden_n), dtype=float)

    def train(self, data_train, target_train, data_test, target_test, limit, learning_rate=0.1):
        train_loss = []
        test_loss = []
        loss = []
        i = 0
        while i < limit:
            if i < 1:
                # 前向传播
                self.predict(data_train)
                loss = self.get_loss(target_train)
                train_loss.append(loss.mean())

                # 误差回传
                self.back_propagation(target_train, learning_rate)
            else:
                self.predict(data_train)
                loss = self.get_loss(target_train)
                if loss.mean() < train_loss[-1]:
                    if (learning_rate * 1.1) < 0.9:
                        learning_rate = 1.1 * learning_rate
                    # else:
                    #     learning_rate = 0.9
                    train_loss.append(loss.mean())
                    self.back_propagation(target_train, learning_rate)
                else:
                    self.back_one()

                    learning_rate = 0.1 * learning_rate
                    i -= 1
                    del(train_loss[-1])
                    del(test_loss[-1])
                    continue
                    # self.predict(target_train)
                    # self.back_propagation(target_train, learning_rate)

            # 对测试集进行预测
            self.predict(data_test)
            loss = self.get_loss(target_test)
            test_loss.append(loss.mean())
            # if i > 55:
            #     print(1)
            # 画出训练误差和测试误差
            if i % 10 == 0 or i == limit - 1:
                print(learning_rate)
                x = np.linspace(1, i + 1, i + 1)
                plt.clf()
                plt.plot(x, train_loss, 'r', label=u'train loss')
                plt.plot(x, test_loss, 'b', label=u'test loss')
                # plt.axis([-1, limit, 0, 0.05])
                plt.legend(loc='best')
                plt.pause(0.01)

            i += 1
        # plt.pause(0)
        # 计算预测准确率
        print('测试集最终预测结果：')
        print(np.column_stack((target_test, self.output_cells, loss)))
        acc = self.get_acc(loss, target_test)
        print('accuracy:%.2f%%' % (100 * acc))


if __name__ == '__main__':
    breast_cancer = datasets.load_breast_cancer()
    dataSet = breast_cancer.data.tolist()
    targets = breast_cancer.target
    targets = targets.reshape(len(targets), 1).tolist()

    dataSet = preprocessing.scale(dataSet)

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
    #
    # dataSet = preprocessing.scale(dataSet)
    # 随机打乱数据集
    temp = list(zip(dataSet, targets))
    np.random.shuffle(temp)
    dataSet, targets = zip(*temp)

    data_train, data_test, target_train, target_test = train_test_split(dataSet, targets,
                                                                        test_size=0.3, random_state=123)
    nn = BPNeuralNetwork(30, 7, 1)

    time_start = time.time()
    nn.train(data_train, target_train, data_test, target_test, 1000, 0.1)
    time_end = time.time()
    print('训练耗时：', time_end - time_start, 's')
    plt.show()

    # time_start = time.time()
    # nn.test(data_test, target_test)
    # time_end = time.time()
    # print('预测耗时：', time_end - time_start, 's')
