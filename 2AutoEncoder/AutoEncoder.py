import gzip
import pickle as cPickle
import numpy as np
import os
import tensorflow as tf

"""
基本的自动编码器
"""


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
    units = [784,400,100,300,784]
    learning_rate = 0.001
    minibatch = 1
    
    # initialize paramters
    w1 = tf.Variable(tf.truncated_normal([units[0],units[1]],stddev = 0.1))
    b1 = tf.Variable(tf.zeros(units[1]))
    
    w2 = tf.Variable(tf.truncated_normal([units[1],units[2]],stddev = 0.1))
    b2 = tf.Variable(tf.zeros(units[2]))
    
    w3 = tf.Variable(tf.truncated_normal([units[2],units[3]],stddev = 0.1))
    b3 = tf.Variable(tf.zeros(units[3]))
    
    w4 = tf.Variable(tf.truncated_normal([units[3],units[4]],stddev = 0.1))
    b4 = tf.Variable(tf.zeros(units[4]))
    
    x = tf.placeholder(tf.float32,[None, units[0]])
    
    hidden1 = tf.nn.relu(tf.matmul(x,w1) + b1)
    hidden2 = tf.nn.relu(tf.matmul(hidden1,w2) + b2)
    hidden3 = tf.nn.relu(tf.matmul(hidden2,w3) + b3)
    
    """
    hidden1 = tf.nn.sigmoid(tf.matmul(x,w1) + b1)
    hidden2 = tf.nn.sigmoid(tf.matmul(hidden1,w2) + b2)
    hidden3 = tf.nn.sigmoid(tf.matmul(hidden2,w3) + b3)
    """
    
#    y = tf.nn.softmax(tf.matmul(hidden1,w2) + b2)
    y = tf.matmul(hidden3,w4) + b4
    answer = tf.matmul(hidden1,w2) + b2
    
    # 真实值
    y_ = tf.placeholder(tf.float32, [None, units[4]])
    # tf.square(y - y_data)
    # loss_total = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    loss_total = tf.reduce_mean(tf.nn.l2_loss(y - y_))
    regularizers = (tf.nn.l2_loss(w1) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(b2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(b3) + tf.nn.l2_loss(w4) + tf.nn.l2_loss(b4))
    # 将正则项加入损失函数
    
    loss_total += (1e-4 * regularizers) # 0.0005
    
    
    p1 = [0.05]*units[2]
    p2 = [0.95]*units[2]
#    sumq = tf.reduce_sum(hidden2, axis=0)    # 0是按列计算
    sumq = tf.reduce_sum(hidden2)
    
#    hidden = hidden2 / sumq
    # ======================= minibatch 增大的时候，这里？ ==========================
    regularizers1 = 5e-5 * tf.nn.softmax_cross_entropy_with_logits(logits=hidden2/sumq, labels=p1)
    regularizers2 = 5e-5 * tf.nn.softmax_cross_entropy_with_logits(logits=1.0-hidden2/sumq, labels=p2)
    loss_total += regularizers1
    loss_total += regularizers2
    """
    loss_total += 5e-4 * tf.reduce_sum( -1.0*hidden * tf.log(hidden / p))
    hidden = 1.0 - hidden
    loss_total += 5e-4 * tf.reduce_sum( -1.0*hidden * tf.log(hidden / (1.0-p)))
    """
    
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)   # 学习率
#    train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entropy)   # 学习率很重要，之前过拟合了

    # 这个测试一下看能不能用
    init_op = tf.global_variables_initializer()
    
    ##############################
    # LEARNING !
    
    print("minibatch = %d" % minibatch)
    
    sess = tf.Session()
    sess.run(init_op)
    # 500000
    for i in range((int)(200000 * 1.0/minibatch)):    # 表示一共选择30W次，5W个样本，就是训练6轮
        start = i % (int(Xtrain.shape[0]/(1.0*minibatch)))
    #    print(start)
        start = start * minibatch
        batch_xs = Xtrain[start:start+minibatch]
        batch_ys = Xtrain[start:start+minibatch]
#        print(sess.run(sumq, feed_dict={x: batch_xs, y_: batch_ys}))
#        print(sess.run(hidden2, feed_dict={x: batch_xs, y_: batch_ys}))
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        if i%1000==0:
            print("step %d " % i)
    
    Xtrain_ = sess.run(answer,feed_dict={x:Xtrain})
    XCV_ = sess.run(answer,feed_dict={x:XCV})
    Xtest_ = sess.run(answer,feed_dict={x: Xtest})
    sess.close()
    #返回
    print("AutoEncoder finish!")
    return Xtrain_,XCV_,Xtest_
    
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
    Xtrain, XCV, Xtest = autoEncoder(Xtrain,XCV,Xtest)
    Xtrain = np.array(Xtrain)
    XCV    = np.array(XCV)
    Xtest  = np.array(Xtest)
    print(Xtrain.shape, XCV.shape, Xtest.shape)
    ##############################
    # sess = tf.InteractiveSession()



