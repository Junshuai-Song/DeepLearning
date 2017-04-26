# Auther: Alan
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


def read_data_flower(input_path, split_path):
    """
    获取train/val/test数据集
    :param input_path: 
    :param split_path: 
    :return: 
    """
    splits = sio.loadmat(split_path)
    train_location = splits["trn1"][0]
    val_location = splits["val1"][0]
    test_location = splits["tst1"][0]

    # print(test_location[0])
    with open(input_path+"/files.txt","r") as f:
        files = f.readlines()

    # 读取train_set
    trainX,valX,testX = [],[],[]
    trainY, valY, testY = [], [], []

    global max_row, max_col
    max_row, max_col = -1,-1
    for i in range(len(files)):
        lena = mpimg.imread(input_path+files[i][:-1])
        x,y = lena.shape[0], lena.shape[1]
        if x > max_row:
            max_row = x
        if y > max_col:
            max_col = y

    box = (0, 0, max_row, max_col)
    print("box: ", box)
    for train in train_location:
        lena = Image.open(input_path + files[int(train)-1][:-1]).crop(box)
        # print(im.size)
        trainX.append(np.reshape(lena, [-1]))
        # region = region.transpose(Image.ROTATE_180)
        # lena = mpimg.imread(input_path + files[int(train)-1][:-1])
        # trainX.append(reshape(lena, max_row, max_col))
        # print(lena[0][0],lena.shape)
        # print(type(lena))
        # one-hot编码
        label = int(float(train) / 80.001)
        trainY.append(get_one_hot(label, 17))
    for val in val_location:
        lena = Image.open(input_path + files[int(train) - 1][:-1]).crop(box)
        valX.append(np.reshape(lena, [-1]))
        # valX.append(np.reshape(lena, [lena.shape[0] * lena.shape[1] * lena.shape[2]]))
        label = int(float(val) / 80.001)
        trainY.append(get_one_hot(label, 17))
    for test in test_location:
        lena = Image.open(input_path + files[int(train) - 1][:-1]).crop(box)
        testX.append(np.reshape(lena, [-1]))
        # testX.append(np.reshape(lena, [lena.shape[0] * lena.shape[1] * lena.shape[2]]))
        label = int(float(test) / 80.001)
        trainY.append(get_one_hot(label, 17))

    # Y改成one-hot编码
    print("read data finish!")
    return np.array(trainX), np.array(trainY), np.array(valX), np.array(valY), np.array(testX), np.array(testY)


def CNN_LeNet_5(input_path, split_path, log_path):
    # LeNet_5的卷积网络，对花分类的数据集进行测试 (500, 541, 3)
    # read_data_flower(input_path, split_path)
    trainX, trainY, valX, valY, testX, testY = read_data_flower(input_path, split_path)
    print("trainX.shape: ",trainX.shape)

    # 构建网络
    x = tf.placeholder(tf.float32, [None, 3465903])
    y_ = tf.placeholder(tf.float32, [None, 17])
    x_image = tf.reshape(x, [-1, 1093, 1057, 3])  # 把向量重新整理成矩阵，最后一个表示通道个数

    # 第一二参数值得卷积核尺寸大小，即patch，第三个参数是图像通道数，第四个参数是卷积核的数目，代表会出现多少个卷积特征
    W_conv1 = weight_variable([10, 10, 3, 32])  # 表示3个通道一起处理，一般方法是加和之后过激活函数
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1) # 547 * 529 * 32


    W_conv2 = weight_variable([10, 10, 32, 64])   # 多通道卷积，卷积出64个特征
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_3x3(h_conv2) # 183 * 177 * 64
    # 参数过多，多加两个max_pool层
    h_pool3 = max_pool_3x3(h_pool2) # 61 * 59
    h_pool4 = max_pool_3x3(h_pool3) # 21 * 20


    W_fc1 = weight_variable([21 * 20 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool4_flat = tf.reshape(h_pool4, [-1, 21 * 20 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 17])
    b_fc2 = bias_variable([17])
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
    summary_writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())

    # for i in range((int)(20000)):
    num_examples = trainX.shape[0]
    minibatch = 128
    for epoch in range(20):
        print("iter:", epoch)
        avg_cost = 0.
        total_batch = int(num_examples / minibatch)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = next_batch(trainX, trainY, minibatch, num_examples)
            # print(type(batch_xs),type(batch_ys))
            # print(batch_xs.shape, batch_ys.shape)

            # and summary nodes
            _, c, summary = sess.run([train_step, cross_entropy, merged_summary_op],feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.8})

            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)
            # Compute average loss
            avg_cost += c / total_batch
            if (i % 1 == 0):
                print("i:", i, "   current c:", c, "   ave_cost:", avg_cost)
            if i%100==0:
                print("val accuracy %g" % sess.run(accuracy, feed_dict={
                    x: valX, y_: valY, keep_prob: 1.0}))
        # Display logs per epoch step
        # if (epoch + 1) % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        # 到达一定程度进行测试test输出
        if epoch % 1 == 0:
            # batchs = mnist.train.next_batch(minibatch)
            print("test accuracy %g" % sess.run(accuracy, feed_dict={
                x: testX, y_: testY, keep_prob: 1.0}))
            # x: batchs[0], y_: batchs[1], keep_prob: 1.0}))

    print("finish!")

def next_batch(trainX, trainY, minibatch, num_examples):
    locations = random.sample([i for i in range(num_examples)], minibatch)
    batch_xs = trainX[locations]
    batch_ys = trainY[locations]
    return batch_xs, batch_ys



if __name__ =="__main__":
    # CNN_LeNet_5_Mnist("./CNN/minist")
    CNN_LeNet_5("./flower_data/jpg/","./flower_data/datasplits.mat","./CNN/flower")



