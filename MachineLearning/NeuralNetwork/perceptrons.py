import numpy as np
import random
import matplotlib.pyplot as plt


# sign function
def sign(v):
    if v > 0:
        return 1
    else:
        return -1


# train function to get weight and bias
def training(data_train1, data_train2):
    # train_data1 = [[1, 1, 1]]  # positive sample
    # train_data2 = [[0, 0, 0], [0, 1, 0], [1, 0, 0]]  # negative sample
    data_train = data_train1 + data_train2

    weight = [0, 0]
    bias = 0
    learning_rate = 0.1

    train_num = int(input("train num:"))

    for i in range(train_num):
        train = random.choice(data_train)
        x1, x2, y = train
        y_predict = sign(weight[0] * x1 + weight[1] * x2 + bias)
        print("train data:x:(%f, %f) y:%d ==>y_predict:%d" % (x1, x2, y, y_predict))
        if y != y_predict:
            weight[0] = weight[0] + learning_rate * (y - y_predict) * x1
            weight[1] = weight[1] + learning_rate * (y - y_predict) * x2
            bias = bias + learning_rate * (y - y_predict)
            print("update weight and bias:")
            print(weight[0], weight[1], bias)
    print("stop training :")
    print(weight[0], weight[1], bias)

    # plot the train data and the hyper curve
    plt.plot(np.array(data_train1)[:, 0], np.array(data_train1)[:, 1], 'ro')
    plt.plot(np.array(data_train2)[:, 0], np.array(data_train2)[:, 1], 'bo')

    x_1 = np.linspace(0, 10, 100)
    x_2 = (-weight[0] * x_1 - bias) / weight[1]
    plt.plot(x_1, x_2)
    plt.axis([-2, 12, -7, 7])
    plt.show()

    return weight, bias


# test function to predict
def load_data_set(file_name):
    data_p = []
    data_n = []
    # data_set = []
    fr = open(file_name)
    for line in fr.readlines():
        line_data = list(map(float, line.strip().split(',')))
        if line_data[2] == 1:
            data_p.append(line_data)
        else:
            data_n.append(line_data)

    return data_p, data_n


def test():
    file_name = 'testSet.txt'
    data_p, data_n = load_data_set(file_name)
    # print(data_p, data_n)
    weight, bias = training(data_p, data_n)
    while True:
        test_data = []

        data = input("enter q to quit,enter test data (x1, x2):")
        if data == 'q':
            break
        test_data += [int(n) for n in data.split(',')]
        predict = sign(weight[0] * test_data[0] + weight[1] * test_data[1] + bias)
        print("predict==>%d" % predict)


if __name__ == "__main__":
    test()
