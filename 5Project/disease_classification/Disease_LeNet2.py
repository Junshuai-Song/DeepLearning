# Auther: Alan
"""
"""
import tensorflow as tf
import random
import os
import scipy.io as sio
# import matplotlib.pyplot as plt # plt
import matplotlib.image as mpimg # mpimg
import numpy as np
# import Image
import math
from PIL import Image
import xlrd


class Disease:
    def __init__(self):
        print("init")

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        # strides
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")


    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    def max_pool_3x3(self, x):
        return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,3,3,1], padding="SAME")


    def max_pool_5x5(self, x):
        return tf.nn.max_pool(x, ksize=[1,5,5,1], strides=[1,5,5,1], padding="SAME")

    def max_pool_10x10(self, x):
        return tf.nn.max_pool(x, ksize=[1,10,10,1], strides=[1,10,10,1], padding="SAME")


    def get_one_hot(self, label, num):
        y = []
        for i in range(num):
            if i == label:
                y.append(1.0)
            else:
                y.append(0.0)
        return y


    def read_data_disease(self, input_file):
        """
        883 747 858
        :return: 
        """
        allX,allY = [],[]
        locations, locations1, locations2 = [], [], []
        with open(input_file + "trainX.txt", "r") as f1, open(input_file + "trainY.txt", "r") as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
            for i in range(len(lines1)):
                x = lines1[i][:-1]
                allX.append(x)
                label = lines2[i][:-1]
                if label=="G0":
                    allY.append(0)
                elif label=="G0-1":
                    allY.append(1)
                else:
                    allY.append(2)

                if x[0:8] == "produce1" and x[-5]=="0":     # add 1
                    locations1.append(i)
                elif x[0:8] == "produce2" and x[-5] == "0": # add 2
                    locations2.append(i)
                elif x[0]!='p':
                    locations.append(i) # normal
        allX = np.array(allX)
        allY = np.array(allY)

        # normal
        print("length of locations: ", len(locations))
        rate = 0.9
        ls = random.sample(locations, int(len(locations)*rate))
        trainX = list(allX[ls])
        trainY = list(allY[ls])

        ls = list(set(locations)-set(ls))
        testX = list(allX[ls])
        testY = list(allY[ls])

        # add 1
        def add_others(ls, trainX, trainY, allX, allY, num):
            for i in range(num):
                l = list(np.array(ls) + i)
                trainX.extend(allX[l])
                trainY.extend(allY[l])
        print("length of locations1: ", len(locations1)*3)
        ls = random.sample(locations1, int(len(locations1)*rate))
        add_others(ls, trainX, trainY, allX, allY, 3)
        ls = list(set(locations1) - set(ls))
        add_others(ls, testX, testY, allX, allY, 3)
        # add 2
        print("length of locations2: ", len(locations2)*6)
        ls = random.sample(locations2, int(len(locations2)*rate))
        add_others(ls, trainX, trainY, allX, allY, 6)
        ls = list(set(locations2) - set(ls))
        add_others(ls, testX, testY, allX, allY, 6)


        trainX = np.array(trainX)
        trainY = np.array(trainY)
        testX = np.array(testX)
        testY = np.array(testY)
        print("shape of all data: ", trainX.shape, trainY.shape, testX.shape, testY.shape)

        cnt, cnt1, cnt2 = 0,0,0
        for i in trainY:
            if i==0:
                cnt+=1
            elif i==1:
                cnt1+=1
            else:
                cnt2+=1
        print("the number of the train data: ", cnt, cnt1, cnt2)

        return trainX, trainY, testX, testY


    def next_batch(self, trainX, trainY, minibatch, num_examples, flag=1):
        locations = random.sample([i for i in range(num_examples)], minibatch)
        batch_xs_ = trainX[locations]
        batch_ys_ = trainY[locations]

        batch_xs, batch_ys = [], []
        # print("minibatch:",minibatch)
        for i in range(batch_xs_.shape[0]):
            # print("i:", i)
            # if i%50==0:
            #     print("deal with: ", i)
            batch = batch_xs_[i]
            # print("file:", batch)
            lena = Image.open("./data/G0/" + batch)
            first, second = 50,50
            if flag==1:
                first = random.sample([j for j in range(100)], 1)[0]
                second = random.sample([j for j in range(100)], 1)[0]
            lena = lena.crop((first, second, first + 600, second + 600))

            # lena.save("temp.jpg")
            # lena = Image.open("temp.jpg")
            # print("new image.size:", lena.size)
            # print("batch_ys_[i]:", batch_ys_[i])
            batch_xs.append(np.reshape(lena, [-1]) / 255.0)
            batch_ys.append(self.get_one_hot(batch_ys_[i], 3))      # one_hot
        return np.array(batch_xs), np.array(batch_ys)

    def get_test(self, testX, testY):
        # get test data in memory.
        batch_xs, batch_ys = [], []
        for i in range(testX.shape[0]):
            if i%50==0:
                print("test i:", i)
            batch = testX[i]
            lena = Image.open("./data/G0/" + batch)
            first, second = 50, 50
            lena = lena.crop((first, second, first + 600, second + 600))
            batch_xs.append(np.reshape(lena, [-1]) / 255.0)
            batch_ys.append(self.get_one_hot(testY[i], 3))      # one_hot
        return np.array(batch_xs), np.array(batch_ys)


    def model(self, input_file, log_path):
        trainX, trainY, testX, testY = self.read_data_disease(input_file)
        cnt = 0
        for i in testY:
            if i == 0:
                cnt += 1
        print("minimal precision: ", float(cnt) / testY.shape[0])


        # 构建网络
        self.x_ = tf.placeholder(tf.float32, [None, 360000 * 3])
        self.y_ = tf.placeholder(tf.float32, [None, 3])
        x_image = tf.reshape(self.x_, [-1, 600, 600, 3])  # 把向量重新整理成矩阵，最后一个表示通道个数

        # 第一二参数值得卷积核尺寸大小，即patch，第三个参数是图像通道数，第四个参数是卷积核的数目，代表会出现多少个卷积特征
        W_conv1 = self.weight_variable([3, 3, 3, 8])
        b_conv1 = self.bias_variable([8])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)  # 300*300

        W_conv2 = self.weight_variable([3, 3, 8, 16])  # 多通道卷积，卷积出64个特征
        b_conv2 = self.bias_variable([16])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_5x5(h_conv2)  # 60*60

        W_conv3 = self.weight_variable([3, 3, 16, 32])  # 多通道卷积，卷积出64个特征
        b_conv3 = self.bias_variable([32])
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = self.max_pool_5x5(h_conv3)  # 12*12

        W_conv4 = self.weight_variable([3, 3, 32, 16])  # 多通道卷积，卷积出64个特征
        b_conv4 = self.bias_variable([16])
        h_conv4 = tf.nn.relu(self.conv2d(h_pool3, W_conv4) + b_conv4)
        h_pool4 = self.max_pool_2x2(h_conv4)  # 6*6

        W_fc1 = self.weight_variable([6 * 6 * 16, 16])
        b_fc1 = self.bias_variable([16])
        h_pool2_flat = tf.reshape(h_pool4, [-1, 6*6*16])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        W_fc2 = self.weight_variable([16, 3])
        b_fc2 = self.bias_variable([3])
        self.y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))   #2/3/4/5
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_conv + 1e-10), reduction_indices=[1]))  # 1
        tf.summary.scalar("cross_entropy", self.cross_entropy)


        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        tf.summary.scalar("learning_rate", self.learning_rate)
        # self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)

        self.correct_prediction = tf.equal(tf.arg_max(self.y_conv, 1), tf.arg_max(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", self.accuracy)

        # init_op = tf.initialize_all_variables()
        init_op = tf.global_variables_initializer()

        self.merged_summary_op = tf.summary.merge_all()

        drops = [1.0]   # overfitting.
        global_step = 101
        for d in range(len(drops)):
            drop = drops[d]
            log_path = log_path + str(d)
            print("log_path: ", log_path, "  drop:", drop)

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

            # sess = tf.Session()
            sess.run(init_op)
            summary_writer_train = tf.summary.FileWriter(log_path+"train", graph=tf.get_default_graph())
            summary_writer_test = tf.summary.FileWriter(log_path+"test", graph=tf.get_default_graph())

            num_examples = trainX.shape[0]
            minibatch = 30
            maxc = -1.0
            for epoch in range(global_step):
                print("iter:", epoch)

                avg_cost = 0.
                total_batch = int(num_examples / minibatch)
                # total_batch = 1

                rate = 0.001 * math.pow(0.7, int(epoch/5))
                print("learning rate: ", rate)

                for i in range(total_batch):
                    batch_xs, batch_ys = self.next_batch(trainX, trainY, minibatch, num_examples)
                    # print(batch_xs.shape, batch_ys.shape)

                    _, c , summary = sess.run([self.train_step, self.cross_entropy, self.merged_summary_op],
                                             feed_dict={self.x_: batch_xs, self.y_: batch_ys, self.learning_rate: rate,
                                                        self.keep_prob: drop})
                    summary_writer_train.add_summary(summary, epoch * total_batch + i)
                    avg_cost += c / total_batch
                    if i%1==0:
                        print("i/tot: ", i, "/", total_batch, "   current c:", c, "   ave_cost:", avg_cost)

                    # # test
                    if i%5==0:
                        # test
                        batch_xs, batch_ys = self.next_batch(testX, testY, minibatch, len(testY), batch=2)
                        pre, summary = sess.run([self.accuracy, self.merged_summary_op],
                                                feed_dict={self.x_: batch_xs, self.y_: batch_ys,
                                                           self.learning_rate: rate,
                                                           self.keep_prob: drop})
                        summary_writer_test.add_summary(summary, epoch * total_batch + i)
                        if pre > maxc:
                            maxc = pre

                # test
                # if epoch % 1 == 0:
                #     batch_xs, batch_ys = self.next_batch_train(testX, testY, minibatch, len(testX))
                #     pre, summary = sess.run([self.accuracy, self.merged_summary_op], feed_dict={self.x_: batch_xs, self.y_: batch_ys, self.learning_rate: rate, self.keep_prob: drop})
                #     summary_writer_test.add_summary(summary, epoch * total_batch)
                #
                #     print("precision: ", pre)
                #     if pre > maxc:
                #         maxc = pre

            print("max precision: ", maxc)


if __name__ =="__main__":
    # CNN_LeNet_5_dev("./cifar_data/train.mat", "./cifar_data/test.mat", "./CNN/cifar")
    disease = Disease()
    disease.model("./data/", "./log_LeNet2_")





