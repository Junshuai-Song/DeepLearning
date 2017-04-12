import gzip
import pickle as cPickle
import numpy as np
import os
import tensorflow as tf

"""
基本的自动编码器
"""

activate_functions = 2
# 优化算法: AdadeltaOptimizer & AdamOptimizer
optimizations = 2
# learning_rates = [0.00001,0.0001,0.001,0.01,0.1,1.0]
# minibatchs = [1,5,25,125,625,3125]

def load_data(dataset):
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join(os.path.split(__file__)[0], "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print('Downloading data from %s' % origin)
        urllib.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f,encoding="bytes")
    f.close()     
    
    return train_set[0], train_set[1], valid_set[0], valid_set[1], test_set[0], test_set[1]
    # train_set[0]: 50000 * 784, train_set[1]: 50000    后面的表示label
    # valid_set[0]: 10000 * 784, valid_set[1]: 10000
    # test_set[0]:  10000 * 784, test_set[1]:  10000
    
def get_batch(label, label_num):
    # the type of label is list.
    ans = []
#    maxc = -1
    for i in range(len(label)):
        t = []
#        if label[i] > maxc:
#            maxc = label[i]
        for j in range(label_num): 
            if label[i]==j:
                t.append(1)
            else:
                t.append(0)
        ans.append(t)
#    print("maxc: %d" % maxc)
    return ans
    
##############################
# read data

def get_mediate():
    intermediate = 100
    return intermediate

def autoEncoder(Xtrain,XCV,Xtest):
    print("AutoEncoder start!")
    # 输入为np.array，返回二维list
    units = [784,300,100,300,784]
    units2 = [784, 300, 100, 40,10]

    learning_rates = [0.001, 0.01, 0.1]
    minibatchs = [5, 50, 500]
    learning_rate1 = 0.005
    minibatch1 = 128

    for learning_rate in learning_rates:
        for activate_function in range(activate_functions):
            for optimization in range(optimizations):
                for minibatch in minibatchs:
                    # initialize paramters
                    w1 = tf.Variable(tf.truncated_normal([units[0],units[1]],stddev = 0.1))
                    b1 = tf.Variable(tf.zeros(units[1]))

                    w2 = tf.Variable(tf.truncated_normal([units[1],units[2]],stddev = 0.1))
                    b2 = tf.Variable(tf.zeros(units[2]))

                    w3 = tf.Variable(tf.truncated_normal([units[2],units[3]],stddev = 0.1))
                    b3 = tf.Variable(tf.zeros(units[3]))

                    w4 = tf.Variable(tf.truncated_normal([units[3],units[4]],stddev = 0.1))
                    b4 = tf.Variable(tf.zeros(units[4]))

                    w33 = tf.Variable(tf.truncated_normal([units2[2], units2[3]], stddev=0.1))
                    b33 = tf.Variable(tf.zeros(units2[3]))

                    w44 = tf.Variable(tf.truncated_normal([units2[3], units2[4]], stddev=0.1))
                    b44 = tf.Variable(tf.zeros(units2[4]))

                    x = tf.placeholder(tf.float32,[None, units[0]])



                    if activate_function == 0:
                        hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)
                        hidden2 = tf.nn.relu(tf.matmul(hidden1, w2) + b2)
                        hidden33 = tf.nn.relu(tf.matmul(hidden2, w33) + b33)
                    else:
                        hidden1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
                        hidden2 = tf.nn.sigmoid(tf.matmul(hidden1, w2) + b2)
                        hidden33 = tf.nn.sigmoid(tf.matmul(hidden2, w33) + b33)
                    hidden3 = tf.nn.relu(tf.matmul(hidden2, w3) + b3)

                #    y = tf.nn.softmax(tf.matmul(hidden1,w2) + b2)
                    y = tf.matmul(hidden3,w4) + b4
                    answer = tf.matmul(hidden1,w2) + b2

                    yy = tf.matmul(hidden33, w44) + b44

                    # 真实值
                    y_ = tf.placeholder(tf.float32, [None, units[4]])
                    y__ = tf.placeholder(tf.float32, [None, units2[4]])
                    # tf.square(y - y_data)

                    # 注意，对于多分类问题，Tensorflow中还没有直接的交叉熵解决方案。 ------------
                    # regularizer = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

                    regularizer = tf.reduce_mean(tf.nn.l2_loss(y - y_)/(1.0*minibatch1))
                    loss_total = regularizer
                    regularizer1 = (tf.nn.l2_loss(w1) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(b2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(b3) + tf.nn.l2_loss(w4) + tf.nn.l2_loss(b4))
                    # 将正则项加入损失函数
                    loss_total += (1e-1 * regularizer1) # 0.0005


                    p1 = tf.constant(0.005)
                #    sumq = tf.reduce_sum(hidden2, axis=0)    # 0是按列计算
                #    sumq = tf.reduce_sum(hidden2)/(units[2]*1.0*minibatch)
                    sumq = tf.reduce_mean(hidden2)
                    regularizer2 = (p1*tf.log(p1/(sumq+1e-8)) + (1.0-p1)*tf.log((1.0-p1)/(1.0-sumq+1e-8)))
                #    hidden = hidden2 / sumq
                    # ======================= minibatch 增大的时候，这里？ ==========================
                #    r1 = hidden2/sumq
                #    r2 = 1.0 - r1
                #    regularizer2 = tf.nn.softmax_cross_entropy_with_logits(logits=sumq, labels=p1)
                    loss_total += (1e-1 * regularizer2)
                    """
                    regularizer2_1 = tf.nn.softmax_cross_entropy_with_logits(logits=hidden2/sumq, labels=p1)
                    regularizer2_2 = tf.nn.softmax_cross_entropy_with_logits(logits=1.0-hidden2/sumq, labels=p2)
                    regularizer2 = tf.add(regularizer2_1, regularizer2_2)
                    loss_total += (1e2 * regularizer2)
                    
                    loss_total += 5e-4 * tf.reduce_sum( -1.0*hidden * tf.log(hidden / p))
                    hidden = 1.0 - hidden
                    loss_total += 5e-4 * tf.reduce_sum( -1.0*hidden * tf.log(hidden / (1.0-p)))
                    """

                    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yy, labels=y__))
                    if optimization == 0:
                        train_step2 = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)  # 学习率
                    else:
                        train_step2 = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entropy)  # 学习率很重要，之前过拟合了

                    # AutoEncoder的
                    train_step = tf.train.AdamOptimizer(learning_rate1).minimize(loss_total)   # 学习率
                #    train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entropy)   # 学习率很重要，之前过拟合了

                    # 这个测试一下看能不能用
                    init_op = tf.global_variables_initializer()

                    correct_prediction = tf.equal(tf.arg_max(yy, 1), tf.arg_max(y__, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


                    ##############################
                    # LEARNING !

                    print("minibatch = %d" % minibatch)

                    sess = tf.Session()

                    # summary_writer = tf.summary.FileWriter('log_simple_stats', sess.graph)

                    sess.run(init_op)
                    # 500000
                    # for i in range((int)(50000000 * 1.0/minibatch)):    # 表示一共选择30W次，5W个样本，就是训练6轮
                    for i in range((int)(10000)):
                        start = i % (int(Xtrain.shape[0]/(1.0*minibatch1)))
                    #    print(start)
                        start = start * minibatch1
                        batch_xs = Xtrain[start:start+minibatch1]
                        batch_ys = Xtrain[start:start+minibatch1]
                #        print(sess.run(sumq, feed_dict={x: batch_xs, y_: batch_ys}))
                #        print(sess.run(hidden2, feed_dict={x: batch_xs, y_: batch_ys}))
                        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

                        if i%1000==0:
                            loss = sess.run(loss_total, feed_dict={x: batch_xs, y_: batch_ys})
                            reg0 = sess.run(regularizer, feed_dict={x: batch_xs, y_: batch_ys})
                            reg1 = sess.run(regularizer1, feed_dict={x: batch_xs, y_: batch_ys})
                            reg2 = sess.run(regularizer2, feed_dict={x: batch_xs, y_: batch_ys})
                            print("step %d, loss %g, reg0 %g, reg1 %g, reg2 %g" % (i, loss, reg0, reg1,reg2))
                    print("AutoEncoder finish!")


                    print(learning_rate, activate_function, optimization, minibatch)
                    # 500000
                    # for i in range((int)(500000 * 1.0/minibatch)):    # 表示一共选择30W次，5W个样本，就是训练6轮
                    for i in range((int)(200000)):
                        start = i % (int(Xtrain.shape[0] / (1.0 * minibatch)))
                        start = start * minibatch
                        #    print(start)
                        batch_xs = Xtrain[start:start + minibatch]
                        batch_ys = ytrain[start:start + minibatch]
                        sess.run(train_step2, feed_dict={x: batch_xs, y__: batch_ys})

                        if i % 1000 == 0:  # 验证集测试
                            #        print(sess.run(cross_entropy,feed_dict={x: XCV, y_: yCV}))
                            cv_accuracy = sess.run(accuracy, feed_dict={x: Xtrain, y__: ytrain})
                            cv_loss = sess.run(cross_entropy, feed_dict={x: Xtrain, y__: ytrain})
                            print("step %d, train accuracy %g, loss %g " % (i, cv_accuracy, cv_loss))
                            cv_accuracy = sess.run(accuracy, feed_dict={x: XCV, y__: yCV})
                            print("step %d, CV    accuracy %g" % (i, cv_accuracy))

                    print("=== test accuracy %g" % sess.run(accuracy, feed_dict={x: Xtest, y__: ytest}))
                    sess.close()


if __name__ == '__main__':
    
    Xtrain, ytrain, XCV, yCV, Xtest, ytest = load_data("mnist.pkl.gz")
    
    # Hopefully numpy is helpful
    Xtrain = np.array(Xtrain)
    XCV    = np.array(XCV)
    Xtest  = np.array(Xtest)
    ytrain = get_batch(ytrain,10)
    yCV = get_batch(yCV,10)
    ytest = get_batch(ytest,10)
    
    
    # Check the sizes of these numpy arrays
    print(Xtrain.shape, XCV.shape, Xtest.shape)
    autoEncoder(Xtrain,XCV,Xtest)
    
    ##############################
    # sess = tf.InteractiveSession()



