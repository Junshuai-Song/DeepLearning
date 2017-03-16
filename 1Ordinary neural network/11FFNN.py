import gzip
import pickle as cPickle
import numpy as np
import os
import tensorflow as tf

"""
Tensorflow
"""

# relu  + AdadeltaOptimizer + 5层

# 激活函数: relu & sigmoid
activate_functions = 2
# 优化算法: AdadeltaOptimizer & AdamOptimizer
optimizations = 2
# 网络结构：设计层数与各层顶点个数
units = [784,300,100,40,10]
# units = [784,300,10]


learning_rates = [0.001,0.01,0.1]
minibatchs = [5,50,500]

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
    
    ##############################
    # sess = tf.InteractiveSession()
    
    for learning_rate in learning_rates:
        for activate_function in range(activate_functions):
            for optimization in range(optimizations):
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
                
                if activate_function==0:
                    hidden1 = tf.nn.relu(tf.matmul(x,w1) + b1)
                    hidden2 = tf.nn.relu(tf.matmul(hidden1,w2) + b2)
                    hidden3 = tf.nn.relu(tf.matmul(hidden2,w3) + b3)
                else: 
                    hidden1 = tf.nn.sigmoid(tf.matmul(x,w1) + b1)
                    hidden2 = tf.nn.sigmoid(tf.matmul(hidden1,w2) + b2)
                    hidden3 = tf.nn.sigmoid(tf.matmul(hidden2,w3) + b3)
                #y = tf.nn.softmax(tf.matmul(hidden1,w2) + b2)
                y = tf.matmul(hidden3,w4) + b4
                
                # 真实值
                y_ = tf.placeholder(tf.float32, [None, units[4]])
                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
                if optimization==0:
                    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)   # 学习率
                else:
                    train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entropy)   # 学习率很重要，之前过拟合了
                # 这个测试一下看能不能用
                init_op = tf.global_variables_initializer()
                
                # tf.local_variables_initializer().run()
                
                # 精度查看
                correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(y_,1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                
                ##############################
                # LEARNING !
                
                # learning_rate = [0.00001,0.0001,0.001,0.01,0.1,1.0]
                # minibatchs
                for minibatch in minibatchs:
                    print(learning_rate,activate_function,optimization,minibatch)
                    
                    sess = tf.Session()
                    # summary_writer = tf.summary.FileWriter('log_simple_stats', sess.graph)
                    
                    sess.run(init_op)
                    # 500000
                    # for i in range((int)(500000 * 1.0/minibatch)):    # 表示一共选择30W次，5W个样本，就是训练6轮
                    for i in range((int)(20000)):   
                        start = i % (int(Xtrain.shape[0]/(1.0*minibatch)))
                        start = start * minibatch
                    #    print(start)
                        batch_xs = Xtrain[start:start+minibatch]
                        batch_ys = ytrain[start:start+minibatch]
                        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
                        
                        if i%1000 == 0:  #验证集测试
                    #        print(sess.run(cross_entropy,feed_dict={x: XCV, y_: yCV}))
                            cv_accuracy = sess.run(accuracy,feed_dict={x:Xtrain, y_: ytrain})
                            cv_loss = sess.run(cross_entropy,feed_dict={x:Xtrain, y_: ytrain})
                            print("step %d, train accuracy %g, loss %g " % (i, cv_accuracy, cv_loss))
                            cv_accuracy = sess.run(accuracy,feed_dict={x:XCV, y_: yCV})
                            print("step %d, CV    accuracy %g" % (i, cv_accuracy))
                    
                    
                    print("=== test accuracy %g" % sess.run(accuracy,feed_dict={x: Xtest, y_: ytest}))
                    sess.close()
        


