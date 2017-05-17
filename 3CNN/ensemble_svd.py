# Auther: Alan
import random
from PIL import Image
from numpy import *
import os
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import numpy as np
from sklearn import svm
from sklearn import metrics

def get_embeddings(bash_path):
    # 从文件中读取每一个照片，使用svd分解，获取其特征值，作为特征
    embeddings = {}
    # box = (0, 0, 100, 100)
    cnt = 0
    for parent, dirnames, filenames in os.walk(bash_path):
        # 对每一个文件夹
        # print(parent,filenames)
        # if parent[11:] not in embeddings:
        # if cnt>200:
        #     break
        if cnt%10==0:
            print("deal with fold number : %d/%d" %(cnt, 5749))
        cnt += 1
        if len(parent) > 11:
            embeddings[parent[11:]] = []
            for filename in filenames:
                # 是图片
                # parent = parent[11:]
                lena = Image.open(parent + "/" + filename)  #统一大小为250*250
                # print(lena.size)
                # lena = Image.open(parent + "/" + filename).crop(box)
                lena = np.reshape(lena, [-1])
                embedding = []
                for i in range(3):
                    data = lena[i*250*250: (i+1)*250*250]
                    data = np.reshape(data, [250,250])
                    u, sigma, vt = linalg.svd(data)
                    embedding.extend(sigma[0:100])  #取前100维
                # 一个样本的特征值
                # print(embedding)
                embeddings[parent[11:]].append(embedding)
    # save(embeddings)
    print("get embeddings!")
    return embeddings

def saveX(file_path, trainX):
    with open(file_path, "w") as f:
        # f.write(str(len(embeddings)) + " 600\n")
        for i in range(len(trainX)):
            f.write(str(i))
            for x in trainX[i]:
                f.write(" " + str(x))
            f.write("\n")
    print("save: " + file_path)

def saveY(file_path, trainY):
    with open(file_path, "w") as f:
        # f.write(str(len(embeddings)) + " 600\n")
        for i in range(len(trainY)):
            f.write(str(trainY[i]) + "\n")

    print("save: " + file_path)


def save_data(trainX, trainY, testX, testY):
    saveX("./trainX", trainX)
    saveY("./trainY", trainY)
    saveX("./testX", testX)
    saveY("./testY", testY)
    print("save data finish!")


def readX(file_path):
    trainX = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            words = line.split(" ")
            words = words[1:]
            x = []
            for word in words:
                x.append(float(word))   # 注意转换回float类型
            trainX.append(x)
    return trainX

def readY(file_path):
    trainY = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            trainY.append(float(line))   # 注意转换回float类型
    return trainY

def read_data():
    trainX = readX("./trainX")
    trainY = readY("./trainY")
    testX = readX("./testX")
    testY = readY("./testY")
    print("read data finish!")
    return np.array(trainX), np.array(trainY), np.array(testX), np.array(testY)


def get_data(bash_path, train_file, embeddings):
    trainX = []
    trainY = []
    with open(bash_path + train_file, "r") as f:
        lines = f.readlines()
        # train_num = int(lines[0][:-1])
        lines = lines[1:]
        # lines = lines[1:10]
        for line in lines:
            words = line.split("\t")
            if len(words)<=3:
                # 表示是正样本
                first = int(words[1])-1
                second = int(words[2][:-1])-1
                x = []
                x.extend(embeddings[words[0]][first])
                x.extend(embeddings[words[0]][second])
                trainX.append(x)
                trainY.append(1.0)
            else:
                # 负样本
                first = int(words[1])-1
                second = int(words[3][:-1])-1
                x = []
                x.extend(embeddings[words[0]][first])
                x.extend(embeddings[words[2]][second])
                trainX.append(x)
                trainY.append(0.0)
    return trainX, trainY

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
def logistic_regression_pre(bash_path, train_file, test_file):
    """
    使用LR进行测试
    :param train_file: 
    :param test_file: 
    :param rate_feature:    特征一共300个，选择比率 
    :param rate_example:    样本选择比率
    :return: 
    """

    trainX, trainY, testX, testY = read_data()

    # clf = svm.SVC(kernel='rbf', gamma=0.7, C=1.0).fit(trainX, trainY)
    # y_predicted = clf.predict(testX)
    W = np.ones(trainX.shape[0]) / trainX.shape[0]

    clf = DecisionTreeClassifier(max_depth=3).fit(trainX, trainY, sample_weight=W)

    # clf = LogisticRegression().fit(trainX, trainY, sample_weight=W)
    y_predicted = clf.predict(testX)
    print("fit finish!")

    print("accuracy: ",metrics.accuracy_score(testY, y_predicted))


    # fraction_of_positives, mean_predicted_value = calibration_curve(testY, prob_pos)
    # print("mean_predicted_value:", mean_predicted_value, fraction_of_positives)


from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


def ensemble(bash_path, train_file, test_file):
    train_x, train_y, test_x, test_y = read_data()

    W = np.ones(train_x.shape[0]) / train_x.shape[0]

    M = 100
    K = 2
    D = 2

    print("D: ",D)

    alpha = np.empty(M)
    g = []

    for m in range(M):
        g.append(DecisionTreeClassifier(max_depth=D))
        g[m].fit(train_x, train_y, sample_weight=W)
        pre_y = g[m].predict(train_x)
        I = (pre_y != train_y)
        e = I.dot(W)
        alpha[m] = np.sqrt((1 - e) * (K - 1) / e)
        W = W * I * alpha[m] + W * -I / alpha[m]
        W = normalize(W.reshape(1, -1), norm='l1').reshape(train_x.shape[0])

    alpha = np.log(alpha)
    pre_y = np.zeros((test_x.shape[0], K))
    accu = np.empty(M)
    for m in range(M):
        print("m:", m)
        pre_y += alpha[m] * g[m].predict_proba(test_x)
        accu[m] = np.sum(np.argmax(pre_y, axis=1) == test_y) / test_x.shape[0]

    print("无提升准确率:\t", accu[0])
    print("最大提升准确率:\t", np.max(accu))
    print("最大提升迭代轮数:\t", np.argmax(accu))

    plt.plot([accu[0]] * M, label="base learner")
    plt.plot(accu, label="my")
    plt.xlabel("iteration")
    plt.ylabel("accuracy rate (%)")
    plt.title("Adaboost (tree max_depth:%d)" % D)
    plt.show()


    print("sklearn...")
    # 使用sklearn自带AdaBoostClassifier进行测试
    accu2 = []
    for m in range(1, M + 1):
        print("m:", m)
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=D), n_estimators=m)
        clf.fit(train_x, train_y)
        pre_y = clf.predict(test_x)
        accu2.append(np.sum(pre_y == test_y) / test_x.shape[0])

    print("Sklearn")
    print("无提升准确率:\t", accu2[0])
    print("最大提升准确率:\t", np.max(accu2))
    print("最大提升迭代轮数:\t", np.argmax(accu2))

    plt.plot(accu2, label="sklearn")
    plt.legend(loc="lower right")
    plt.xlabel("iteration")
    plt.ylabel("accuracy rate (%)")
    plt.title("Adaboost (tree max_depth:%d)" % D)
    plt.show()



if __name__ =="__main__":
    # 特征：SVD
    # 模型：LR
    bash_path = "./lfw_data/"
    train_file = "_pairsDevTrain.txt"
    test_file = "_pairsDevTest.txt"

    # embeddings = get_embeddings(bash_path)
    # trainX, trainY = get_data(bash_path, train_file, embeddings)
    # testX, testY = get_data(bash_path, test_file, embeddings)
    # print("get train and test data!")
    #
    # # 保存训练和测试数据 --- 之后直接读取
    # save_data(trainX, trainY, testX, testY)
    # print("save train and test data!")

    # logistic_regression_pre(bash_path, train_file, test_file)
    ensemble(bash_path, train_file, test_file)




