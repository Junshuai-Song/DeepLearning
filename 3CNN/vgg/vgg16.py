########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from imagenet_classes import class_names
from PIL import Image


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial)

def bias_variable(shape):
    # 这里完全可以用一个数组代替 tf.zeros(units[1])
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)

class vgg16:
    def __init__(self,x, imgs, weights=None, sess=None):
        self.x = x
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc3l)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)
            print("load weights.")


    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        self.y_ = tf.placeholder(tf.float32, [None, 100])

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)

            self.fc3 = tf.nn.relu(self.fc3l)        # 自己增加的
            self.parameters += [fc3w, fc3b]

        # 自己多加一层，但是不加到parameters中，不用
        with tf.name_scope('fc4') as scope:
            fc4w = weight_variable([1000, 100])
            fc4b = bias_variable([100])

            fc4l = tf.nn.bias_add(tf.matmul(self.fc3, fc4w), fc4b)
            self.y_conv = tf.nn.softmax(fc4l)

        # 增加loss
        with tf.name_scope('loss') as loss:
            self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_conv + 1e-10), reduction_indices=[1]))  # 1
            self.train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(self.cross_entropy)  # 5
            # self.train_step = tf.train.MomentumOptimizer(0.1, 0.9).minimize(self.cross_entropy)

        correct_prediction = tf.equal(tf.arg_max(self.y_conv, 1), tf.arg_max(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))

from scipy.io import loadmat
import random
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
    return train_x, train_y, test_x, test_y


def next_batch(trainX, trainY, minibatch, num_examples):
    # 这里需要多一步，将样本扩充到224*224 （）
    locations = random.sample([i for i in range(num_examples)], minibatch)
    batch_xs_ = trainX[locations]
    batch_ys = trainY[locations]
    for j in range(batch_xs_.shape[0]):
        batch = batch_xs_[j]
        # lena = Image.fromarray(np.reshape(batch, [32,32,3]))
        lena = Image.fromarray(np.reshape(np.reshape(batch, [3, 1024]).T, [32, 32, 3]))
        lena.save("./temp/temp_cifar" +str(j)+ ".png")
    # new_img = Image.new('RGB', (224, 224), 255)
    return

    #
    # vgg16 这里有错误！！！！！

    # 方法1：复制
    batch_xs = []
    for batch in batch_xs_:
        ans = []
        for j in range(3):
            for k in range(7):
                # print(j, k, len(ans))
                for i in range(32):
                    t = batch[i*32+j*1024:(i+1)*32+j*1024]
                    t = list(t)
                    ans.extend(t*7) #扩充7倍刚好
        batch_xs.append(ans)

    # 方法2：直接scale扩充

    batch_xs = np.array(batch_xs)
    return batch_xs, batch_ys

def next_batch_test(trainX, trainY, minibatch, num_examples, i):
    # 这里需要多一步，将样本扩充到224*224 （）
    locations = [j for j in range(50 * i, 50 * (i + 1))]
    batch_xs_ = trainX[locations]
    batch_ys = trainY[locations]

    # 方法1：复制
    batch_xs = []
    for batch in batch_xs_:
        ans = []
        for j in range(3):
            for k in range(7):
                # print(j, k, len(ans))
                for i in range(32):
                    t = batch[i*32+j*1024:(i+1)*32+j*1024]
                    t = list(t)
                    ans.extend(t*7) #扩充7倍刚好
        batch_xs.append(ans)

    # 方法2：直接scale扩充

    batch_xs = np.array(batch_xs)
    return batch_xs, batch_ys

def vgg_test():
    # 测试vgg
    sess = tf.Session()

    x = tf.placeholder(tf.float32, [None, 50176 * 3])
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(x, imgs, 'vgg16_weights.npz', sess)


    img1 = imread('laska.png', mode='RGB')
    img1 = imresize(img1, (224, 224))

    prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print(class_names[p], prob[p])

def print_accuracy(vgg, sess, testX, testY):
    # 数据集量比较大，每轮分开测试
    num = int(10000/50)
    tot = 0.0
    for i in range(num):
        x,y = next_batch_test(testX, testY, 50, 10000, i)
        acc = sess.run(vgg.accuracy, feed_dict={vgg.x: x, vgg.y_: y})
        tot += acc
        if (num % 1 == 0):
            print("test number(50): ", i, "  acc: ", acc)
    tot /= num
    return tot  # tot是平均精度

def vgg_cifar():
    sess = tf.Session()

    x = tf.placeholder(tf.float32, [None, 50176 * 3])
    imgs = tf.reshape(x, [-1, 224, 224, 3])
    # vgg = vgg16(x, imgs, 'vgg16_weights.npz', sess)

    # 数据集
    trainX, trainY, testX, testY = read_data_cifar("../cifar_data/train.mat", "../cifar_data/test.mat")
    print("trainX.shape: ", trainX.shape, trainY.shape, testX.shape, testY.shape)
    # testX, testY = next_batch(testX, testY, 10, 10000)
    next_batch(trainX, trainY, 10, 10000)
    return

    # # 开始训练
    num_examples = trainX.shape[0]
    minibatch = 128
    minibatch = 48
    maxc = -1.0
    global_step = 100
    for epoch in range(global_step):
        # 到达一定程度进行测试test输出

        print("iter:", epoch)
        avg_cost = 0.
        total_batch = int(num_examples / minibatch)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = next_batch(trainX, trainY, minibatch, num_examples)

            _, c = sess.run([vgg.train_step, vgg.cross_entropy], feed_dict={vgg.x: batch_xs, vgg.y_: batch_ys})

            avg_cost += c / total_batch
            if (i % 1 == 0):
                print("i:", i, "   current c:", c, "   ave_cost:", avg_cost)

        # Display logs per epoch step
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        if epoch % 1 == 0:
            # batchs = mnist.train.next_batch(minibatch)
            acc = print_accuracy(vgg, sess, testX, testY)
            if acc > maxc:
                maxc = acc
            print("test accuracy %g" % acc)
            # x: batchs[0], y_: batchs[1], keep_prob: 1.0}))
        print("====================================================================")


    print("max acc: ", maxc)


    sess.close()

if __name__ == '__main__':

    # vgg_test()
    vgg_cifar()

