import numpy as np
from dataset.store import loadCSV, saveXLSX
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
import math
from datetime import datetime
from libs.roc import getScore
from sklearn import metrics

folder = 'dataset/3 classes/iris'
file = 'iris.data'

# ====================================================

data = np.asarray(loadCSV(folder + '/' + file))

y, X = data[:, 4], data[:, :4].astype(float)

y[y == 'Iris-setosa'] = 0
y[y == 'Iris-versicolor'] = 1
y[y == 'Iris-virginica'] = 2

y = np.asarray([[1, 0, 0] if (_ == '0') else [0, 1, 0] if (_ == '1') else [0, 0, 1] for _ in y])

# ====================================================
learning_rate = [0.001, 0.005, 0.01, 0.05]
learning_rule = ['adagrad', 'sgd']
hidden_units = [8, 16, 32, 64, 128, 256, 512]
hidden_layers = [2, 3, 4]
n_iters = [16, 32, 64, 128, 256, 512, 1024]
# ====================================================

def initWeight(size):
    return tf.Variable(tf.zeros(size))

def initWeight_U(size):
    minR, maxR = 0, 0
    if (len(size) == 1):
        minR, maxR = -1/size[0], 1/size[0]
    elif (len(size) == 2):
        _ = math.sqrt(size[0] + size[1])
        minR, maxR = -1 / _, 1 / _

    return tf.Variable(tf.random_uniform(size, minR, maxR))

output = {}
output['score'] = [['learning_rate', 'learning_rule', 'hidden_units', 'n_iters', '#', 'acc', 'u', 'VUS_1', 'VUS_2', 'TP1', 'F12', 'F13', 'F21', 'TP2', 'F23', 'F31', 'F32', 'TP3' 'countingtime']]


for lrt in learning_rate:
    for lr in learning_rule:
        for hu in hidden_units:
            for ni in n_iters:
                for hl in hidden_layers:
                    for ii in range(2):
                        print('learning_rate = ', lrt, 'learning_rule = ', lr, 'hidden_units = ', hu, 'n_iters = ', ni,
                              '## = ', ii)
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

                        x = tf.placeholder(tf.float32, [None, X_train.shape[1]])

                        _initWeight = initWeight_U
                        W, b, a = [], [], []

                        W.append(_initWeight([X_train.shape[1], hu]))
                        b.append(_initWeight([hu]))
                        a.append(tf.nn.sigmoid(tf.matmul(x, W[-1]) + b[-1]))

                        for j in range(hl-1):
                            W.append(_initWeight([hu, hu]))
                            b.append(_initWeight([hu]))
                            a.append(tf.nn.sigmoid(tf.matmul(a[-1], W[-1]) + b[-1]))

                        W.append(_initWeight([hu, y_train.shape[1]]))
                        b.append(_initWeight([y_train.shape[1]]))

                        yy = tf.nn.softmax(tf.matmul(a[-1], W[-1]) + b[-1])
                        yy_ = tf.placeholder(tf.float32, [None, y_train.shape[1]])

                        cross_entropy = tf.reduce_mean(-tf.reduce_sum(yy_ * tf.log(yy), reduction_indices=[1]))

                        if (lr == 'adagrad'):
                            train_step = tf.train.AdagradOptimizer(lrt).minimize(cross_entropy)
                        else:
                            train_step = tf.train.GradientDescentOptimizer(lrt).minimize(cross_entropy)

                        init = tf.initialize_all_variables()

                        sess = tf.Session()
                        sess.run(init)


                        startTime = datetime.now()
                        for i in range(ni):
                            # print(i)
                            batch_xs, batch_ys = X_train, y_train
                            sess.run(train_step, feed_dict={x: batch_xs, yy_: batch_ys})
                        endTime = datetime.now()

                        correct_prediction = tf.equal(tf.argmax(yy,1), tf.argmax(yy_,1))
                        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                        y_score = sess.run(yy, feed_dict={x: X_test, yy_: y_test})
                        acc = sess.run(accuracy, feed_dict={x: X_test, yy_: y_test})
                        y_hat = np.argmax(y_score, axis=1)
                        y_test = np.argmax(y_test, axis=1)

                        confMatrix = confusion_matrix(y_test, y_hat).tolist()

                        tmp = np.append(X_test, np.reshape(y_test, (1, y_test.shape[0])).T, axis=1)
                        tmp = np.append(tmp, np.reshape(y_hat, (1, y_hat.shape[0])).T, axis=1)
                        tmp = np.append(tmp, y_score, axis=1)

                        output['data'] = [['X_' + str(i) for i in range(1, X_test.shape[1] + 1)] +
                                          ['y_label', 'y_hat', 'y_score', 'y_score']] + \
                                         tmp.tolist()

                        score_u = getScore('U', y_test, y_score)
                        score_VUS_1 = getScore('VUS_1', y_test, y_hat)
                        score_VUS_2 = getScore('VUS_2', y_test, y_score)

                        output['score'].append(
                            [lrt, lr, hu, ni, ii, acc, score_u, score_VUS_1, score_VUS_2]
                            + confMatrix[0] + confMatrix[1] + confMatrix[2] +
                            [(endTime - startTime).seconds + (endTime - startTime).microseconds / 10 ** 6])

saveXLSX(output, 'tensor_dl_3classes_tmp.xlsx')