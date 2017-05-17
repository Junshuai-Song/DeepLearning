# Auther: Alan
"""
    实现AlexNet网络结构：但是实际上AlexNet网络结构的餐率领有60M，其实际的参数量比后来的几个网络结构都要多，这里不选择
    但是尝试实现更深层的卷积网络来查看性能
        这里一共整理了4层
"""

# Auther: Alan
"""
将LeNet5应用在Cifar数据集上
"""
import tensorflow as tf
import random
import os
import scipy.io as sio
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
# import Image
from PIL import Image

global max_row, max_col

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # 这里完全可以用一个数组代替 tf.zeros(units[1])
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # strides表示每一维度的步长
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
    # ksize表示池化窗口的大小, 其中最前面的1和最后的1分别表示batch和channel（这里不考虑对不同batch做池化，所以设置为1）
    # 另外一个任务：判断两张图片是否为同一个人，觉得可以将其当做不同channel，一起进行池化的操作
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,3,3,1], padding="SAME")


def max_pool_5x5(x):
    return tf.nn.max_pool(x, ksize=[1,5,5,1], strides=[1,5,5,1], padding="SAME")


def CNN_LeNet_5_Mnist(logs_path):
    """
    LeNet对Mnist数据集进行测试
    :return: 
    """
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # print(mnist)

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    x_image = tf.reshape(x, [-1,28,28,1])   # 把向量重新整理成矩阵，最后一个表示通道个数

    # 第一二参数值得卷积核尺寸大小，即patch，第三个参数是图像通道数，第四个参数是卷积核的数目，代表会出现多少个卷积特征
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


    W_conv2 = weight_variable([5, 5, 32, 64])   # 多通道卷积，卷积出64个特征
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    tf.summary.scalar("cross_entropy", cross_entropy)

    correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    merged_summary_op = tf.summary.merge_all()
    # 初始化变量
    init_op = tf.global_variables_initializer()

    # 开始训练
    sess = tf.Session()
    sess.run(init_op)
    # iterate
    # Xtrain, ytrain = get_batch(self.args, self.simrank, self.walks, minibatch * 100, self.tem_simrank)  # 找一个大点的数据集测试效果
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # for i in range((int)(20000)):
    num_examples = 12800*2    #这里暂时手动设置吧
    minibatch = 128
    for epoch in range(20):
        print("iter:", epoch)
        avg_cost = 0.
        total_batch = int(num_examples / minibatch)
        # Loop over all batches
        for i in range(total_batch):
            batchs = mnist.train.next_batch(minibatch)
            batch_xs, batch_ys = batchs[0], batchs[1]
            # batch_xs, batch_ys = next_batch(self.args, self.simrank, self.walks, minibatch, self.tem_simrank,
            #                                 num_examples)

            # and summary nodes
            _, c, summary = sess.run([train_step, cross_entropy, merged_summary_op], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)
            # Compute average loss
            avg_cost += c / total_batch
            if (i % 10 == 0):
                print("i:", i, "   current c:", c, "   ave_cost:", avg_cost)
        # Display logs per epoch step
        # if (epoch + 1) % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        # 到达一定程度进行测试test输出
        if epoch%1==0:
            batchs = mnist.train.next_batch(minibatch)
            print("test accuracy %g" % sess.run(accuracy, feed_dict={
                x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
                # x: batchs[0], y_: batchs[1], keep_prob: 1.0}))

def get_one_hot(label, num):
    y = []
    for i in range(num):  # 一共17个类别
        if i == label:
            y.append(1.0)
        else:
            y.append(0.0)
    return y


from scipy.io import loadmat

def read_data_cifar(train_file, test_file):
    """
    获取train/val/test数据集
    :param input_path: 
    :param split_path: 
    :return: 
    """
    f1 = loadmat(train_file)
    f2 = loadmat(test_file)
    train_x = f1["data"]
    train_y_ = f1["fine_labels"]
    test_x = f2["data"]
    test_y_ = f2["fine_labels"]

    # 需要处理labels
    train_y = []
    for train in train_y_:
        y = []
        for i in range(100):
            if i == int(train)-1:
                y.append(1.0)
            else:
                y.append(0.0)
        train_y.append(y)

    test_y = []
    for test in test_y_:
        y = []
        for i in range(100):
            if i == int(test) - 1:
                y.append(1.0)
            else:
                y.append(0.0)
        test_y.append(y)

    train_y = np.array(train_y)
    test_y = np.array(test_y)

    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    return train_x/255.0, train_y, test_x/255.0, test_y


def CNN_LeNet_5_dev(train_file, test_file, log_path):
    trainX, trainY, testX, testY = read_data_cifar(train_file, test_file)
    print("trainX.shape: ", trainX.shape, trainY.shape, testX.shape, testY.shape)

    # 构建网络
    x = tf.placeholder(tf.float32, [None, 1024*3])
    y_ = tf.placeholder(tf.float32, [None, 100])
    x_image = tf.reshape(x, [-1,32,32,3])   # 把向量重新整理成矩阵，最后一个表示通道个数

    # 第一二参数值得卷积核尺寸大小，即patch，第三个参数是图像通道数，第四个参数是卷积核的数目，代表会出现多少个卷积特征
    W_conv1 = weight_variable([3, 3, 3, 64])
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)     # 16*16


    W_conv2 = weight_variable([3, 3, 64, 64])   # 多通道卷积，卷积出64个特征
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)     # 8*8

    W_conv3 = weight_variable([3, 3, 64, 128])  # 多通道卷积，卷积出64个特征
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)     # 4*4

    W_conv4 = weight_variable([3, 3, 128, 128])  # 多通道卷积，卷积出64个特征
    b_conv4 = bias_variable([128])
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4)     # 2*2


    W_fc1 = weight_variable([2*2*128, 2*128])
    b_fc1 = bias_variable([2*128])
    h_pool2_flat = tf.reshape(h_pool4, [-1, 2*2*128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([2*128, 100])
    b_fc2 = bias_variable([100])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))   #2/3/4/5
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv+1e-10), reduction_indices=[1]))  #1


    # train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)    #1/2
    train_step = tf.train.GradientDescentOptimizer(1e-5).minimize(cross_entropy)    # 3/4
    train_step = tf.train.GradientDescentOptimizer(1e-1).minimize(cross_entropy)  # 5
    train_step_3 = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)

    # tf.summary.scalar("cross_entropy", cross_entropy)

    correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    merged_summary_op = tf.summary.merge_all()
    # 初始化变量
    init_op = tf.global_variables_initializer()
    # summary_writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())

    # 开始训练
    drops = [1.0, 0.8, 0.6, 0.4, 0.2]
    drops = [0.4]
    for i in range(len(drops)):
        drop = drops[i]
        log_path = log_path + str(i)
        print("log_path: ", log_path, "  drop:", drop)


        sess = tf.Session()
        sess.run(init_op)
        # iterate
        # Xtrain, ytrain = get_batch(self.args, self.simrank, self.walks, minibatch * 100, self.tem_simrank)  # 找一个大点的数据集测试效果

        # for i in range((int)(20000)):
        num_examples = trainX.shape[0]
        minibatch = 128
        maxc = -1.0
        for epoch in range(1000):
            print("iter:", epoch)
            if epoch > 800:
                train_step = train_step_3

            avg_cost = 0.
            total_batch = int(num_examples / minibatch)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = next_batch(trainX, trainY, minibatch, num_examples)
                # print(type(batch_xs),type(batch_ys))
                # print(batch_xs.shape, batch_ys.shape)
                # print(batch_xs[0])

                # and summary nodes
                # print(sess.run(h_pool4, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.8}))
                # print(sess.run(y_conv, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.8}))
                # print(sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.8}))
                # return
                # _, c, summary = sess.run([train_step, cross_entropy, merged_summary_op],feed_dict={x: batch_xs, y_: batch_ys, keep_prob: drop})
                _, c = sess.run([train_step, cross_entropy],
                                         feed_dict={x: batch_xs, y_: batch_ys, keep_prob: drop})

                # Write logs at every iteration
                # summary_writer.add_summary(summary, epoch * total_batch + i)
                # Compute average loss
                avg_cost += c / total_batch
                if (i % 1 == 0):
                    print("i:", i, "   current c:", c, "   ave_cost:", avg_cost)

                # if i % 500 == 0:
                #     # batchs = mnist.train.next_batch(minibatch)
                #     print("test accuracy %g" % sess.run(accuracy, feed_dict={
                #         x: testX, y_: testY, keep_prob: 1.0}))

            # Display logs per epoch step
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

            # 到达一定程度进行测试test输出
            if epoch % 1 == 0:
                # batchs = mnist.train.next_batch(minibatch)
                acc = sess.run(accuracy, feed_dict={x: testX, y_: testY, keep_prob: 1.0})
                if acc > maxc:
                    maxc = acc
                print("test accuracy %g" % acc)
                # x: batchs[0], y_: batchs[1], keep_prob: 1.0}))
            print("====================================================================")
        sess.close()
        print("max acc: ", maxc)
        print("finish!")
    print("finish all!")


def next_batch(trainX, trainY, minibatch, num_examples):
    locations = random.sample([i for i in range(num_examples)], minibatch)
    batch_xs = trainX[locations]
    batch_ys = trainY[locations]
    return batch_xs, batch_ys



if __name__ =="__main__":
    # 尝试对LeNet网络加深结构，到5层卷积，尝试效果，这里使用默认的dropout比例0.4
    CNN_LeNet_5_dev("./cifar_data/train.mat", "./cifar_data/test.mat", "./CNN/cifar")




