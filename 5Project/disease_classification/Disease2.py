# Auther: Alan
"""
还没试，因为之前Disease.py没收敛

现在好像都收敛不了，尝试下VGG吧。
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


    def next_batch(self, trainX, trainY, minibatch, num_examples):
        locations = random.sample([i for i in range(num_examples)], minibatch)
        batch_xs_ = trainX[locations]
        batch_ys_ = trainY[locations]

        batch_xs, batch_ys = [], []
        # print("minibatch:",minibatch)
        for i in range(batch_xs_.shape[0]):
            # print("i:", i)
            if i%50==0:
                print("deal with: ", i)
            batch = batch_xs_[i]
            # print("file:", batch)
            lena = Image.open("./data/G0/" + batch)

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

    def my_image_filter(self, x_image):
        self.h_conv1 = tf.nn.relu(self.conv2d(x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = self.max_pool_2x2(self.h_conv1)  # 300*300*32
        self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = self.max_pool_5x5(self.h_conv2)  # 60*60*32
        self.h_conv3 = tf.nn.relu(self.conv2d(self.h_pool2, self.W_conv3) + self.b_conv3)

        self.h_conv4 = tf.nn.relu(self.conv2d(self.h_conv3, self.W_conv4_attr) + self.b_conv4_attr)
        self.h_pool4 = self.max_pool_2x2(self.h_conv4)  # 30*30*64
        self.h_conv5 = tf.nn.relu(self.conv2d(self.h_pool4, self.W_conv5_attr) + self.b_conv5_attr)
        self.h_pool5 = self.max_pool_5x5(self.h_conv5)  # 6*6
        self.h_flat1 = tf.reshape(self.h_pool5, [-1, 6 * 6 * 8])

        self.h_flat1_drop = tf.nn.dropout(self.h_flat1, self.keep_prob)
        self.h_fc1 = tf.matmul(self.h_flat1_drop, self.W_fc1_attr) + self.b_fc1_attr

        self.h_fc_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)
        self.y_conv = tf.matmul(self.h_fc_drop, self.W_fc2) + self.b_fc2

        return self.y_conv


    def model(self, input_file, log_path):
        trainX, trainY, testX, testY = self.read_data_disease(input_file)
        cnt = 0
        for i in testY:
            if i == 0:
                cnt += 1
        print("minimal precision: ", float(cnt) / testY.shape[0])
        # testX,testY = self.get_test(testX, testY, testX.shape[0], testY.shape[0])


        self.keep_prob = tf.placeholder(tf.float32)
        self.x_ = tf.placeholder(tf.float32, [None, 600 * 600 * 3])  # shape [None, 128]
        self.x = tf.reshape(self.x_, [-1, 600, 600, 3])

        self.W_conv1 = self.weight_variable([5, 5, 3, 8])
        self.b_conv1 = self.bias_variable([8])
        self.W_conv2 = self.weight_variable([3, 3, 8, 32])
        self.b_conv2 = self.bias_variable([32])
        self.W_conv3 = self.weight_variable([5, 5, 32, 32])
        self.b_conv3 = self.bias_variable([32])

        # attr
        self.W_conv4_attr = self.weight_variable([3, 3, 32, 16])
        self.b_conv4_attr = self.bias_variable([16])
        self.W_conv5_attr = self.weight_variable([3, 3, 16, 8])
        self.b_conv5_attr = self.bias_variable([8])
        self.W_fc1_attr = self.weight_variable([6 * 6 * 8, 36])  # 288 * 36
        self.b_fc1_attr = self.bias_variable([36])

        self.W_fc2 = self.weight_variable([36, 3])
        self.b_fc2 = self.bias_variable([3])


        self.y_ = tf.placeholder(tf.float32, [None, 3])     # shape [None, 3]
        self.y_conv = tf.nn.softmax(self.my_image_filter(self.x))

        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_conv + 1e-10), reduction_indices=[1]))
        # self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_conv, labels=self.y_))
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
        global_step = 501
        for i in range(len(drops)):
            drop = drops[i]
            log_path = log_path + str(i)
            print("log_path: ", log_path, "  drop:", drop)

            sess = tf.Session()
            sess.run(init_op)
            summary_writer_train = tf.summary.FileWriter(log_path + "train", graph=tf.get_default_graph())
            summary_writer_test = tf.summary.FileWriter(log_path + "test", graph=tf.get_default_graph())

            num_examples = trainX.shape[0]
            minibatch = 30
            maxc = -1.0
            for epoch in range(global_step):
                print("iter:", epoch)

                avg_cost = 0.
                total_batch = int(num_examples / minibatch)
                # total_batch = 1

                rate = 0.00001 * math.pow(0.7, int(epoch/40))
                print("learning rate: ", rate)

                for i in range(total_batch):
                    batch_xs, batch_ys = self.next_batch(trainX, trainY, minibatch, num_examples)

                    # _, c, summary = sess.run([self.train_step, self.cross_entropy, self.merged_summary_op], feed_dict={self.x_: batch_xs, self.y_: batch_ys, self.learning_rate: rate, self.keep_prob: drop})
                    _, c , summary = sess.run([self.train_step, self.cross_entropy, self.merged_summary_op],
                                             feed_dict={self.x_: batch_xs, self.y_: batch_ys, self.learning_rate: rate,
                                                        self.keep_prob: drop})
                    summary_writer_train.add_summary(summary, epoch * total_batch + i)
                    avg_cost += c / total_batch
                    if i%1==0:
                        print("i/tot: ", i, "/", total_batch, "   current c:", c, "   ave_cost:", avg_cost)

                    # test
                    batch_xs, batch_ys = self.next_batch(testX, testY, minibatch, len(testY))
                    # batch_xs, batch_ys = self.next_batch_train(testX, testY, minibatch, len(testX))
                    pre, summary = sess.run([self.accuracy, self.merged_summary_op],
                                            feed_dict={self.x_: batch_xs, self.y_: batch_ys, self.learning_rate: rate,
                                                       self.keep_prob: drop})
                    summary_writer_test.add_summary(summary, epoch * total_batch + i)
                    if pre > maxc:
                        maxc = pre


                # if epoch % 5 == 0 and epoch > 40:
                #     pre, summary = sess.run([self.accuracy, self.merged_summary_op], feed_dict={self.x_: testX, self.y_: testY, self.keep_prob: drop})
                #     summary_writer.add_summary(summary, epoch * total_batch)
                #
                #     print("precision: ", pre)
                #     if pre > maxc:
                #         maxc = pre

            print("max precision: ", maxc)


if __name__ =="__main__":
    # CNN_LeNet_5_dev("./cifar_data/train.mat", "./cifar_data/test.mat", "./CNN/cifar")
    disease = Disease()
    disease.model("./data/", "./log_2_")





