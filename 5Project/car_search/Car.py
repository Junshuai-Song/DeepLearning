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


class Car:
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


    def read_data_car(self, train_file, test_file):
        """
        :param input_path: 
        :param split_path: 
        :return: 
        """
        self.train_file = train_file
        self.test_file = test_file
        trainX,trainY, train_model = [],[],[]
        with open(train_file+"trainX.txt", "r") as f1, open(train_file+"trainY.txt", "r") as f2, open(train_file + "train_model.txt", "r") as f3:
            lines = f1.readlines()
            for line in lines:
                trainX.append(line[:-1])
            lines = f2.readlines()
            for line in lines:
                trainY.append(line[:-1])
            lines = f3.readlines()
            for line in lines:
                train_model.append(line[:-1])
        trainX = np.array(trainX)
        trainY = np.array(trainY)
        train_model = np.array(train_model)

        locations = random.sample([i for i in range(trainX.shape[0])], int(trainX.shape[0]*0.05))
        valX, valY, val_model = trainX[locations], trainY[locations], train_model[locations]

        train_set = set([i for i in range(trainX.shape[0])])
        val_set = set(locations)
        locations = list(train_set - val_set)
        trainX, trainY, train_model = trainX[locations], trainY[locations], train_model[locations]

        # test
        testX = []
        with open(test_file+"test.txt","r") as f1:
            lines = f1.readlines()
            for line in lines:
                testX.append(line[:-1])

        testX_all = []
        with open(test_file+"test_all.txt","r") as f1:
            lines = f1.readlines()
            for line in lines:
                testX_all.append(line[:-1])

        return trainX, trainY, train_model, valX, valY, val_model, np.array(testX), np.array(testX_all)


    def deal_data_car(self, trainX, trainY, train_model):
        # print(train_model[0:100])
        id_images = [[] for i in range(40000)]
        id_models = [[] for i in range(40000)]

        ids = [0]*40000
        for i in range(trainY.shape[0]):
            local = int(trainY[i])
            ids[local] += 1
            id_images[local].append(trainX[i])
            id_models[local].append(int(train_model[i]))

        # change to np.array
        for i in range(len(id_images)):
            id_images[i] = np.array(id_images[i])
            id_models[i] = np.array(id_models[i])
        return np.array(ids), np.array(id_images), np.array(id_models)


    def get_true(self, batch_xs_, batch_ys_):
        batch_xs, batch_ys = [], []
        for i in range(batch_xs_.shape[0]):
            batch = batch_xs_[i]
            lena = Image.open(self.train_file + "images/" + batch + ".jpg")
            width = lena.size[0]
            height = lena.size[1]
            first = width - 400
            second = height - 400
            first = random.sample([j for j in range(max(first, 1))], 1)[0]
            second = random.sample([j for j in range(max(second, 1))], 1)[0]
            # print("first, second:", first, second)
            lena = lena.crop((first, second, first + 400, second + 400))
            # lena.save("temp.jpg")
            # lena = Image.open("temp.jpg")
            # print("new image.size:", lena.size)
            batch_xs.append(np.reshape(lena, [-1]) / 255.0)
            batch_ys.append(self.get_one_hot(batch_ys_[i], 250))

        return np.array(batch_xs), np.array(batch_ys)

    def get_true_test(self, batch_xs_):
        batch_xs = []
        for i in range(batch_xs_.shape[0]):
            batch = batch_xs_[i]
            lena = Image.open(self.test_file + "images/" + batch + ".jpg")
            width = lena.size[0]
            height = lena.size[1]
            first = width - 400
            second = height - 400
            first = random.sample([j for j in range(max(first, 1))], 1)[0]
            second = random.sample([j for j in range(max(second, 1))], 1)[0]
            # print("first, second:", first, second)
            lena = lena.crop((first, second, first + 400, second + 400))
            batch_xs.append(np.reshape(lena, [-1]) / 255.0)

        return np.array(batch_xs)

    def next_batch(self, trainY, ids, id_images, id_models, minibatch, num_examples):
        locations_set = trainY[random.sample([i for i in range(num_examples)], 100)]
        locations_set = set(locations_set)
        locations = random.sample(list(locations_set), 2)
        first = int(locations[0])
        second = int(locations[1])
        # print("first, second:", first, second)
        # print("first:", first, ids[first])
        locations = random.sample([i for i in range(ids[first])] * minibatch, minibatch)
        batch_xs_ = id_images[first][locations]
        batch_ys_ = id_models[first][locations]


        locations = random.sample([i for i in range(ids[first])] * minibatch, minibatch)    # re-sample
        batch_po_xs_ = id_images[first][locations]
        batch_po_ys_ = id_models[first][locations]

        locations = random.sample([i for i in range(ids[second])] * minibatch, minibatch)  # re-sample
        batch_ne_xs_ = id_images[second][locations]
        batch_ne_ys_ = id_models[second][locations]
        # print()

        # get true batch from images.
        batch_xs, batch_ys = self.get_true(batch_xs_, batch_ys_)
        # print("len:", len(batch_xs_), len(batch_xs_[0]))
        # print("len:", len(batch_xs), len(batch_xs[0]), batch_xs.shape)
        # for id in id_models:
        #     if(id.shape[0]>0):
        #         print("id.shape: ", id.shape)
        #         for j in id:
        #             print(j)
        # print(batch_ys_[0:3], batch_ys[0:3])
        batch_po_xs, batch_po_ys = self.get_true(batch_po_xs_, batch_po_ys_)
        batch_ne_xs, batch_ne_ys = self.get_true(batch_ne_xs_, batch_ne_ys_)

        return batch_xs, batch_ys, batch_po_xs, batch_po_ys, batch_ne_xs, batch_ne_ys

    def my_image_filter(self, x_image):
        self.h_conv1 = tf.nn.relu(self.conv2d(x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = self.max_pool_2x2(self.h_conv1)  # 200*200*32
        # print(self.h_pool1, self.W_conv2)
        # self.conv2d(self.h_pool1, self.W_conv2)
        self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = self.max_pool_5x5(self.h_conv2)  # 40*40*64
        self.h_conv3 = tf.nn.relu(self.conv2d(self.h_pool2, self.W_conv3) + self.b_conv3)

        # attribute
        self.h_conv4_attr = tf.nn.relu(self.conv2d(self.h_conv3, self.W_conv4_attr) + self.b_conv4_attr)
        self.h_pool4 = self.max_pool_2x2(self.h_conv4_attr)  # 20*20*64
        self.h_conv5_attr = tf.nn.relu(self.conv2d(self.h_pool4, self.W_conv5_attr) + self.b_conv5_attr)
        self.h_pool5_attr = self.max_pool_5x5(self.h_conv5_attr)  # 4*4
        self.h_flat1_attr = tf.reshape(self.h_pool5_attr, [-1, 4 * 4 * 32])

        self.h_flat1_attr_drop = tf.nn.dropout(self.h_flat1_attr, self.keep_prob)
        self.h_fc1_attr = tf.matmul(self.h_flat1_attr_drop, self.W_fc1_attr) + self.b_fc1_attr

        # coupled
        self.h_conv4_coup = tf.nn.relu(self.conv2d(self.h_conv3, self.W_conv4_coup) + self.b_conv4_coup)
        self.h_pool4_coup = self.max_pool_2x2(self.h_conv4_coup)  # 20*20*64
        self.h_conv5_coup = tf.nn.relu(self.conv2d(self.h_pool4_coup, self.W_conv5_coup) + self.b_conv5_coup)
        self.h_pool5_coup = self.max_pool_5x5(self.h_conv5_coup)  # 4*4
        self.h_flat1_coup = tf.reshape(self.h_pool5_coup, [-1, 4 * 4 * 32])
        self.h_flat1_coup_drop = tf.nn.dropout(self.h_flat1_coup, self.keep_prob)       # 0.5
        self.h_fc1_coup = tf.matmul(self.h_flat1_coup_drop, self.W_fc1_coup) + self.b_fc1_coup

        return self.h_fc1_attr, self.h_fc1_coup

    def last_layer(self, h_fc):
        self.h_fc_drop = tf.nn.dropout(h_fc, self.keep_prob)
        self.y_conv = tf.matmul(self.h_fc_drop, self.W_fc2) + self.b_fc2
        return self.y_conv

    def model(self, train_file, test_file, log_path):

        trainX, trainY, train_model, valX, valY, val_model, testX, testX_all = self.read_data_car(train_file, test_file)
        print("trainX.shape: ", trainX.shape, valX.shape, testX.shape)
        ids, id_images, id_models = self.deal_data_car(trainX, trainY, train_model)
        ids_val, id_images_val, id_models_val = self.deal_data_car(valX, valY, val_model)
        # return

        self.keep_prob = tf.placeholder(tf.float32)
        self.anchors = tf.placeholder(tf.float32, [None, 400 * 400 * 3])  # shape [None, 128]
        self.positives = tf.placeholder(tf.float32, [None, 400 * 400 * 3])  # shape [None, 128]
        self.negatives = tf.placeholder(tf.float32, [None, 400 * 400 * 3])  # shape [None, 128]
        self.anchors_image = tf.reshape(self.anchors, [-1, 400, 400, 3])
        self.positives_image = tf.reshape(self.positives, [-1, 400, 400, 3])
        self.negatives_image = tf.reshape(self.negatives, [-1, 400, 400, 3])

        # variables_dict = {
        #     "W_conv1": tf.Variable(tf.random_normal([3, 3, 3, 32]),name="W_conv1"),
        #     "b_conv1": tf.Variable(tf.zeros([32]), name="b_conv1"),
        #     "W_conv2": tf.Variable(tf.random_normal([3, 3, 32, 64]),name="W_conv2"),
        #     "b_conv2": tf.Variable(tf.zeros([64]), name="b_conv2"),
        #     "W_conv3": tf.Variable(tf.random_normal([5,5,64,64]),name="W_conv3"),
        #     "b_conv3": tf.Variable(tf.zeros([64]), name="b_conv3"),
        #
        #     "W_conv4_attr": tf.Variable(tf.random_normal([3,3,64,64]),name="W_conv4_attr"),
        #     "b_conv4_attr": tf.Variable(tf.zeros([64]), name="b_conv4_attr"),
        #     "W_conv5_attr": tf.Variable(tf.random_normal([3,3,64,32]),name="W_conv5_attr"),
        #     "b_conv5_attr": tf.Variable(tf.zeros([32]), name="b_conv5_attr"),
        #     "W_fc1_attr": tf.Variable(tf.random_normal([4*4*32, 250]),name="W_fc1_attr"), # 512 * 250
        #     "b_fc1_attr": tf.Variable(tf.zeros([250]), name="b_fc1_attr"),
        #
        #     "W_conv4_coup": tf.Variable(tf.random_normal([3,3,64,64]),name="W_conv4_coup"),
        #     "b_conv4_coup": tf.Variable(tf.zeros([64]), name="b_conv4_coup"),
        #     "W_conv5_coup": tf.Variable(tf.random_normal([3,3,64,32]),name="W_conv5_coup"),
        #     "b_conv5_coup": tf.Variable(tf.zeros([32]), name="b_conv5_coup"),
        #     "W_fc1_coup": tf.Variable(tf.random_normal([4*4*32, 250]),name="W_fc1_coup"),  # 512 * 250
        #     "b_fc1_coup": tf.Variable(tf.zeros([250]), name="b_fc1_coup"),
        #
        #     "W_fc2": tf.Variable(tf.random_normal([250 * 2, 128]),name="W_fc2"),
        #     "b_fc2": tf.Variable(tf.zeros([128]), name="b_fc2")
        #
        # }

        self.W_conv1= self.weight_variable([3, 3, 3, 32])
        self.b_conv1= self.bias_variable([32])
        self.W_conv2= self.weight_variable([3, 3, 32, 64])
        self.b_conv2= self.bias_variable([64])
        self.W_conv3= self.weight_variable([5, 5, 64, 64])
        self.b_conv3= self.bias_variable([64])

        self.W_conv4_attr= self.weight_variable([3, 3, 64, 64])
        self.b_conv4_attr= self.bias_variable([64])
        self.W_conv5_attr= self.weight_variable([3, 3, 64, 32])
        self.b_conv5_attr= self.bias_variable([32])
        self.W_fc1_attr= self.weight_variable([4 * 4 * 32, 250])  # 512 * 250
        self.b_fc1_attr= self.bias_variable([250])

        self.W_conv4_coup= self.weight_variable([3, 3, 64, 64])
        self.b_conv4_coup= self.bias_variable([64])
        self.W_conv5_coup= self.weight_variable([3, 3, 64, 32])
        self.b_conv5_coup= self.bias_variable([32])
        self.W_fc1_coup= self.weight_variable([4 * 4 * 32, 250])  # 512 * 250
        self.b_fc1_coup= self.bias_variable([250])

        self.W_fc2= self.weight_variable([250 * 2, 128])
        self.b_fc2= self.bias_variable([128])


        self.anchors_label = tf.placeholder(tf.float32, [None, 250])     # shape [None, 250]
        self.positives_label = tf.placeholder(tf.float32, [None, 250])   # shape [None, 250]
        self.negatives_label = tf.placeholder(tf.float32, [None, 250])   # shape [None, 250]

        self.anchors_attr_, self.anchors_coup_ = self.my_image_filter(self.anchors_image)
        self.positives_attr_, self.positives_coup_ = self.my_image_filter(self.positives_image)
        self.negatives_attr_, self.negatives_coup_ = self.my_image_filter(self.negatives_image)

        self.anchors_attr = tf.nn.softmax(self.anchors_attr_)
        self.positives_attr = tf.nn.softmax(self.positives_attr_)
        self.negatives_attr = tf.nn.softmax(self.negatives_attr_)
        # attr loss
        # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv + 1e-10), reduction_indices=[1]))
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.anchors_label * tf.log(self.anchors_attr + 1e-10), reduction_indices=[1]))
                        # + \
                        # tf.reduce_mean(-tf.reduce_sum(positives_label * tf.log(positives_attr + 1e-10), reduction_indices=[1])) + \
                        # tf.reduce_mean(-tf.reduce_sum(negatives_label * tf.log(negatives_attr + 1e-10), reduction_indices=[1]))
        # self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.anchors_attr, labels=self.anchors_label))

        self.loss = 0.5 * self.cross_entropy

        self.margin = 0.2
        # triple loss
        self.anchors_coup = tf.nn.relu(self.anchors_coup_)
        self.positives_coup = tf.nn.relu(self.positives_coup_)
        self.negatives_coup = tf.nn.relu(self.negatives_coup_)

        self.t_pos = tf.reduce_sum(tf.square(self.anchors_coup - self.positives_coup), 1)
        self.t_neg = tf.reduce_sum(tf.square(self.anchors_coup - self.negatives_coup), 1)

        self.triple_loss = tf.reduce_mean(tf.maximum(0.0, self.margin + self.t_pos - self.t_neg))
        self.loss += 0.5 * self.triple_loss

        # merge
        self.h_fc1 = tf.concat(1, [self.anchors_attr, self.anchors_coup])
        self.h_fc2 = tf.concat(1, [self.positives_attr, self.positives_coup])
        self.h_fc3 = tf.concat(1, [self.negatives_attr, self.negatives_coup])


        self.y_conv1 = tf.nn.relu(self.last_layer(self.h_fc1))
        self.y_conv2 = tf.nn.relu(self.last_layer(self.h_fc2))
        self.y_conv3 = tf.nn.relu(self.last_layer(self.h_fc3))

        self.d_pos = tf.reduce_sum(tf.square(self.y_conv1 - self.y_conv2), 1)
        self.d_neg = tf.reduce_sum(tf.square(self.y_conv1 - self.y_conv3), 1)

        self.all_loss = tf.reduce_mean(tf.maximum(0.0, self.margin + self.d_pos - self.d_neg))
        self.loss += self.all_loss




        # train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        # self.train_step = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum = 0.9).minimize(self.cross_entropy)
        # train_step = tf.train.GradientDescentOptimizer(10.0).minimize(cross_entropy)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_attribute, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # merged_summary_op = tf.summary.merge_all()
        init_op = tf.initialize_all_variables()
        # summary_writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())

        drops = [0.5]
        global_step = 5
        for i in range(len(drops)):
            drop = drops[i]
            log_path = log_path + str(i)
            print("log_path: ", log_path, "  drop:", drop)


            sess = tf.Session()
            sess.run(init_op)

            num_examples = trainX.shape[0]
            minibatch = 5
            maxc = -1.0
            for epoch in range(global_step):
                print("iter:", epoch)

                avg_cost = 0.
                total_batch = int(num_examples / minibatch)

                rate = 0.01 * math.pow(0.7, int(epoch))
                print("learning rate: ", rate)

                # Loop over all batches
                # batch_xs, batch_ys, batch_po_xs, batch_po_ys, batch_ne_xs, batch_ne_ys = self.next_batch(trainY,
                #                                                                                          ids,
                #                                                                                          id_images,
                #                                                                                          id_models,
                #                                                                                          minibatch,
                for i in range(total_batch):
                    batch_xs, batch_ys, batch_po_xs, batch_po_ys, batch_ne_xs, batch_ne_ys = self.next_batch(trainY,
                                                                                                             ids,
                                                                                                             id_images,
                                                                                                             id_models,
                                                                                                             minibatch,
                                                                                                             num_examples)

                    # ====================
                    # print("batch shape:", batch_xs.shape, batch_ys.shape, batch_po_xs.shape, batch_po_ys.shape, batch_ne_xs.shape, batch_ne_ys.shape)
                    # t = [1.0+k/400.0/400.0 for k in range(400*400*3)]
                    # batch_xs = np.array([t for j in range(5)])
                    # t = [1.0 for k in range(250)]
                    # batch_ys = np.array([t for j in range(5)])
                    # print("get data.")


                    # and summary nodes
                    # _, c = sess.run([train_step, loss], feed_dict={x: batch_xs, y_attribute: batch_ys, step: float(int(epoch/8000.0))})

                    # pos,neg, _, cr, tr, all = sess.run([self.t_pos, self.t_neg, self.train_step, self.cross_entropy, self.triple_loss, self.all_loss], feed_dict={self.anchors: batch_xs, self.anchors_label: batch_ys,self.positives: batch_po_xs,self.positives_label: batch_po_ys,self.negatives: batch_ne_xs,self.negatives_label: batch_ne_ys,self.learning_rate: rate, self.keep_prob: drop})
                    _, c = sess.run(
                        [self.train_step, self.loss],
                        feed_dict={self.anchors: batch_xs, self.anchors_label: batch_ys, self.positives: batch_po_xs,
                                   self.positives_label: batch_po_ys, self.negatives: batch_ne_xs,
                                   self.negatives_label: batch_ne_ys, self.learning_rate: rate, self.keep_prob: drop})

                    # print("pos", pos)
                    # print("neg", neg)
                    # Write logs at every iteration
                    # summary_writer.add_summary(summary, epoch * total_batch + i)
                    # Compute average loss
                    # c = cr + tr + all
                    avg_cost += c / total_batch
                    if i%10==0:
                        print("i:", i, "   current c:", c, "   ave_cost:", avg_cost)

                # Display logs per epoch step
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

                # validate
                if epoch%1==0 and epoch>0:
                    # saver_path = saver.save(sess, "save/save" + epoch + "/model.ckpt")
                    # saver_path = saver.save(sess, "save/save" + epoch + "/model.ckpt")
                    # print("Model saved in file:", saver_path)
                    embedding_trainx = []
                    embedding_trainy = []
                    embedding_testx = []
                    embedding_testy = []

                    for i in range(id_images.shape[0]):
                    # for i in range(100):
                        if i%100==0:
                            print("train embedding: ", i)
                        for j in range(id_images[i].shape[0]):
                            batch_xs_ = np.array([id_images[i][j]])
                            batch_ys_ = np.array([0 for k in range(250)])
                            batch_xs, batch_ys = self.get_true(batch_xs_, batch_ys_)

                            embedding = sess.run([self.y_conv1], feed_dict={self.anchors: batch_xs, self.keep_prob: 1.0})
                            embedding_trainx.append(embedding[0])
                            embedding_trainy.append(i)
                            # print("train insert...")

                    for i in range(id_images_val.shape[0]):
                    # for i in range(100):
                        for j in range(id_images_val[i].shape[0]):
                            batch_xs_ = np.array([id_images_val[i][j]])
                            batch_ys_ = np.array([0 for k in range(250)])
                            batch_xs, batch_ys = self.get_true(batch_xs_, batch_ys_)

                            embedding = sess.run([self.y_conv1], feed_dict={self.anchors: batch_xs, self.keep_prob: 1.0})
                            embedding_testx.append(embedding[0])
                            embedding_testy.append(i)
                            # print("test insert...")
                    pre = self.precision(embedding_trainx, embedding_trainy, embedding_testx, embedding_testy)
                    if pre > maxc:
                        maxc = pre

            print("max precision: ", maxc)


            # test: get all the 128 embedding!
            print("get each answer of the test set!")
            # saver_path = saver.save(sess, "save/save_final/model.ckpt")
            # print("Model saved in file:", saver_path)

            embedding_trainx = []
            embedding_testx = []
            for i in range(testX_all.shape[0]):
            # for i in range(21):
                if i%100==0:
                    print("test embedding number:", i)
                batch_xs_ = np.array([testX_all[i]])
                batch_xs = self.get_true_test(batch_xs_)
                embedding = sess.run([self.y_conv1], feed_dict={self.anchors: batch_xs, self.keep_prob: 1.0})
                embedding_trainx.append(embedding[0])


            # for e in range(5):  # run 5 times
            for e in range(1):  # run 5 times
                print("test run times: ", e)
                for i in range(testX.shape[0]):
                # for i in range(21):
                    batch_xs_ = np.array([testX[i]])
                    batch_xs = self.get_true_test(batch_xs_)
                    embedding = sess.run([self.y_conv1], feed_dict={self.anchors: batch_xs, self.keep_prob: 1.0})
                    embedding_testx.append(embedding[0])

            self.process_answer(testX_all, embedding_trainx, embedding_testx)
            """
            sess.close()
            print("finish!")

            # test =
            # train_ans = []
            # test_ans = []
            # for i in range(train.shape[0]):
            #     embedding = sess.run([y_conv1], feed_dict={anchors: batch_ys})
            #     train_ans.append()
            # for i in range(test.shape[0]):
            #     embedding = sess.run([y_conv1], feed_dict={anchors: batch_ys})
            """


    def precision(self, trainX, trainY, testX, testY):
        # calculate the precision on the val test.
        topk = 5    # test the top 5.
        tot = 0.0
        for i in range(len(testX)):
            if i%100==0:
                print("test number: ", i)
            test = testX[i]
            ans = []
            sim = []
            for j in range(len(trainX)):
                train = trainX[j]
                sim.append(np.sum(np.array(test)*np.array(train)))
            for j in range(topk):
                minc = min(sim)
                flag = sim.index(minc)
                ans.append(flag)
                sim[flag] = 1e8

            cnt = 0
            for j in range(topk):
                if ans[j]==testY[i]:
                    cnt+=1
            tot += cnt
        pre = tot/len(testX)/topk
        print("top 5, precision: ", pre)
        return pre

    def process_answer(self, testX_all, trainX, testX):
        topk = 200  # test the top 200.
        tot = 0.0
        finals = []
        for i in range(len(testX)):
            if i % 100 == 0:
                print("test number: ", i)
            test = testX[i]
            ans = []
            sim = []
            for j in range(len(trainX)):
                train = trainX[j]
                sim.append(np.sum(np.array(test) * np.array(train)))
            for j in range(topk):
                minc = min(sim)
                flag = sim.index(minc)
                ans.append(testX_all[flag])
                sim[flag] = 1e8
            finals.append(ans)

        with open("./answer.txt", "w") as f:
            for k in range(len(finals)):
                ans = finals[k]
                # print(ans)
                for i in range(len(ans)):
                    f.write(ans[i] + " ")
                f.write("\n")




    def test_next_batch(self, train_file, test_file, log_path):

        trainX, trainY, train_model, valX, valY, val_model, testX = self.read_data_car(train_file, test_file)
        print("trainX.shape: ", trainX.shape, valX.shape, testX.shape)
        ids, id_images, id_models = self.deal_data_car(trainX, trainY, train_model)
        # return

        num_examples = trainX.shape[0]
        minibatch = 5

        batch_xs, batch_ys, batch_po_xs, batch_po_ys, batch_ne_xs, batch_ne_ys = self.next_batch(trainY,
                                                                                                         ids,
                                                                                                         id_images,
                                                                                                         id_models,
                                                                                                         minibatch,
                                                                                                         num_examples)
        # print(batch_ys[0:2])
        print("batch shape:", batch_xs.shape, batch_ys.shape, batch_po_xs.shape, batch_po_ys.shape, batch_ne_xs.shape,
              batch_ne_ys.shape)


if __name__ =="__main__":
    # CNN_LeNet_5_dev("./cifar_data/train.mat", "./cifar_data/test.mat", "./CNN/cifar")
    car = Car()
    # car.test_next_batch("./data/train/", "./data/val/", "./CNN/cifar")
    car.model("./data/train/", "./data/val/", "./CNN/cifar")





