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


    def read_data_disease(self, train_file):
        """
        :param input_path: 
        :param split_path: 
        :return: 
        """
        self.train_file = train_file
        trainX,trainY = [],[]

        data = xlrd.open_workbook(train_file)
        table = data.sheets()[0]
        first = table.col_values(0)
        second = table.col_values(1)
        # print("length of first,second:", len(first), len(second))
        for i in range(len(second)):
            label = second[i]
            if label == 'null' or label=='':
                continue
            if label == 'G0':
                trainY.append(0)
            else:
                trainY.append(1)
                # trainY.append(1)    # double
                # trainX.append(first[i])
            trainX.append(first[i])

        locations = set([i for i in range(len(trainX))])
        locations_train = set(random.sample([i for i in range(len(trainX))], int(len(trainX)*0.8)))
        locations_test = list(locations - locations_train)
        locations_train = list(locations_train)

        # cnt = 0
        # for i in trainY:
        #     if i==1:
        #         cnt+=1
        # print("cnt:", cnt, len(trainY))

        trainX = np.array(trainX)
        trainY = np.array(trainY)

        testX = trainX[locations_test]
        testY = trainY[locations_test]
        trainX = trainX[locations_train]
        trainY = trainY[locations_train]


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

            # first = random.sample([j for j in range(100)], 1)[0]
            # second = random.sample([j for j in range(100)], 1)[0]
            # lena = lena.crop((first, second, first + 600, second + 600))

            # ~ add train samples
            """
            rotate_rate = random.sample([j for j in range(100)], 1)[0]
            if rotate_rate<= 25:
                lena = lena
            elif rotate_rate <=50:
                lena = lena.rotate(90)
            elif rotate_rate <= 75:
                lena = lena.rotate(180)
            else:
                lena = lena.rotate(270)
            """
            # lena.save("temp.jpg")
            # lena = Image.open("temp.jpg")
            # print("new image.size:", lena.size)
            # print("batch_ys_[i]:", batch_ys_[i])
            batch_xs.append(np.reshape(lena, [-1]) / 255.0)
            batch_ys.append(self.get_one_hot(batch_ys_[i], 2))
        return np.array(batch_xs), np.array(batch_ys)

    def next_batch_train(self, trainX, trainY, minibatch, num_examples):
        l_0,l_1 = [], []
        for i in range(trainY.shape[0]):
            if trainY[i][0]>0.5:
                l_0.append(i)
            else:
                l_1.append(i)
        # print(l_0, l_1)
        locations = random.sample(l_0, int(minibatch/2.0))
        locations1 = random.sample(l_1, int(minibatch / 2.0))
        # print("locations: ", locations1)

        locations.extend(locations1)
        batch_xs = trainX[locations]
        batch_ys = trainY[locations]
        return np.array(batch_xs), np.array(batch_ys)

    def my_image_filter(self, x_image):
        self.h_conv1 = tf.nn.relu(self.conv2d(x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = self.max_pool_2x2(self.h_conv1)  # 350*350*32
        self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = self.max_pool_5x5(self.h_conv2)  # 70*70*32
        self.h_conv3 = tf.nn.relu(self.conv2d(self.h_pool2, self.W_conv3) + self.b_conv3)

        # attribute
        self.h_conv4 = tf.nn.relu(self.conv2d(self.h_conv3, self.W_conv4_attr) + self.b_conv4_attr)
        self.h_pool4 = self.max_pool_2x2(self.h_conv4)  # 35*35*64
        self.h_conv5 = tf.nn.relu(self.conv2d(self.h_pool4, self.W_conv5_attr) + self.b_conv5_attr)
        self.h_pool5 = self.max_pool_5x5(self.h_conv5)  # 7*7
        self.h_flat1 = tf.reshape(self.h_pool5, [-1, 7 * 7 * 8])

        self.h_flat1_drop = tf.nn.dropout(self.h_flat1, self.keep_prob)
        self.h_fc1 = tf.matmul(self.h_flat1_drop, self.W_fc1_attr) + self.b_fc1_attr


        self.h_fc_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)
        self.y_conv = tf.matmul(self.h_fc_drop, self.W_fc2) + self.b_fc2

        return self.y_conv


    def model(self, train_file, log_path):
        trainX, trainY, testX, testY = self.read_data_disease(train_file)
        print("trainX.shape: ", trainX.shape, trainY.shape, testX.shape, testY.shape)
        trainX, trainY = self.next_batch(trainX, trainY, trainX.shape[0], trainX.shape[0])
        cnt = 0
        for i in testY:
            if i == 0:
                cnt += 1
        print("minimal precision: ", float(cnt) / testY.shape[0])
        testX,testY = self.next_batch(testX, testY, testX.shape[0], testY.shape[0])


        self.keep_prob = tf.placeholder(tf.float32)
        self.x_ = tf.placeholder(tf.float32, [None, 700 * 700 * 3])  # shape [None, 128]
        self.x = tf.reshape(self.x_, [-1, 700, 700, 3])

        self.W_conv1 = self.weight_variable([5, 5, 3, 32])
        self.b_conv1 = self.bias_variable([32])
        self.W_conv2 = self.weight_variable([3, 3, 32, 32])
        self.b_conv2 = self.bias_variable([32])
        self.W_conv3 = self.weight_variable([5, 5, 32, 32])
        self.b_conv3 = self.bias_variable([32])

        # attr
        self.W_conv4_attr = self.weight_variable([3, 3, 32, 16])
        self.b_conv4_attr = self.bias_variable([16])
        self.W_conv5_attr = self.weight_variable([3, 3, 16, 8])
        self.b_conv5_attr = self.bias_variable([8])
        self.W_fc1_attr = self.weight_variable([7 * 7 * 8, 36])  # 288 * 36
        self.b_fc1_attr = self.bias_variable([36])


        self.W_fc2 = self.weight_variable([36, 2])
        self.b_fc2 = self.bias_variable([2])


        self.y_ = tf.placeholder(tf.float32, [None, 2])     # shape [None, 250]
        self.y_conv = tf.nn.softmax(self.my_image_filter(self.x))

        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_conv + 1e-10), reduction_indices=[1]))
        # self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_conv, labels=self.y_))

        tf.summary.scalar("cross_entropy", self.cross_entropy)


        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        tf.summary.scalar("learning_rate", self.learning_rate)
        # self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)

        self.pre_correct_sum = self.y_conv
        self.pre_correct_sum1 = tf.arg_max(self.y_conv, 1)
        self.pre_correct_sum2 = tf.arg_max(self.y_, 1)
        self.correct_prediction = tf.equal(tf.arg_max(self.y_conv, 1), tf.arg_max(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", self.accuracy)

        init_op = tf.initialize_all_variables()
        # self.merged_summary_op = tf.merge_all_summaries()



        drops = [1.0]
        global_step = 101
        for i in range(len(drops)):
            drop = drops[i]
            log_path = log_path + str(i)
            print("log_path: ", log_path, "  drop:", drop)

            sess = tf.Session()
            # summary_writer = tf.train.SummaryWriter(log_path, sess.graph)
            sess.run(init_op)

            num_examples = trainX.shape[0]
            minibatch = 30
            maxc = -1.0
            for epoch in range(global_step):
                print("iter:", epoch)

                avg_cost = 0.
                total_batch = int(num_examples / minibatch)
                # total_batch = 1

                rate = 0.00001 * math.pow(0.7, int(epoch/10))
                print("learning rate: ", rate)

                for i in range(total_batch):
                    batch_xs, batch_ys = self.next_batch_train(trainX, trainY, minibatch, num_examples)
                    print(batch_xs.shape, batch_ys.shape)

                    # _, c, summary = sess.run([self.train_step, self.cross_entropy, self.merged_summary_op], feed_dict={self.x_: batch_xs, self.y_: batch_ys, self.learning_rate: rate, self.keep_prob: drop})
                    _, c = sess.run([self.train_step, self.cross_entropy],
                                             feed_dict={self.x_: batch_xs, self.y_: batch_ys, self.learning_rate: rate,
                                                        self.keep_prob: drop})
                    # summary_writer.add_summary(summary, epoch * total_batch + i)

                    avg_cost += c / total_batch
                    if i%1==0:
                        print("i/tot: ", i, "/", total_batch, "   current c:", c, "   ave_cost:", avg_cost)

                # Display logs per epoch step
                # print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

                    # validate
                    # if epoch%1==0:
                    if i % 3 == 0:
                        pre, s, s1, s2 = sess.run([self.accuracy, self.pre_correct_sum, self.pre_correct_sum1, self.pre_correct_sum2], feed_dict={self.x_: testX, self.y_: testY, self.keep_prob: drop})
                        print("precision: ", pre, "  sum:", s, s1, s2)
                        if pre > maxc:
                            maxc = pre

                # test acc on the train test to see ..!
                trainX_set, trainY_set = self.next_batch_train(trainX, trainY, int(trainX.shape[0]/10), trainX.shape[0])
                pre, s, s1, s2 = sess.run([self.accuracy, self.pre_correct_sum, self.pre_correct_sum1, self.pre_correct_sum2], feed_dict={self.x_: trainX_set, self.y_: trainY_set, self.keep_prob: drop})
                print("precision on train set(1/10): ", pre, "   precision correct sum: ", s1, s2)

            print("max precision: ", maxc)


if __name__ =="__main__":
    # CNN_LeNet_5_dev("./cifar_data/train.mat", "./cifar_data/test.mat", "./CNN/cifar")
    disease = Disease()
    disease.model("./data/label.xlsx", "./log/disease")





