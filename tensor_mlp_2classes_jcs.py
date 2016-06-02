import numpy as np
from dataset.store import loadCSV, saveXLSX
from sklearn.cross_validation import train_test_split
import tensorflow as tf
import math
from datetime import datetime
from libs.roc import getScore
from sklearn import metrics

folder = 'dataset/2 classes/japanese_create_screening'
file = 'crx.csv'

# ====================================================

data = np.asarray(loadCSV(folder + '/' + file))

data[data[:, 0] == '-', 0] = 0
data[data[:, 0] == '+', 0] = 1

data[data[:, 1] == 'a', 1] = 0
data[data[:, 1] == 'b', 1] = 1
data[data[:, 1] == '?', 1] = -1

data[data[:, 2] == '?', 2] = -1
data[data[:, 3] == '?', 3] = -1

data[data[:, 4] == 'u', 4] = 0
data[data[:, 4] == 'y', 4] = 1
data[data[:, 4] == 'l', 4] = 2
data[data[:, 4] == 't', 4] = 3
data[data[:, 4] == '?', 4] = -1

data[data[:, 5] == 'g', 5] = 0
data[data[:, 5] == 'p', 5] = 1
data[data[:, 5] == 'gg', 5] = 2
data[data[:, 5] == '?', 5] = -1

data[data[:, 6] == 'c', 6] = 0
data[data[:, 6] == 'd', 6] = 1
data[data[:, 6] == 'cc', 6] = 2
data[data[:, 6] == 'i', 6] = 3
data[data[:, 6] == 'j', 6] = 4
data[data[:, 6] == 'k', 6] = 5
data[data[:, 6] == 'm', 6] = 6
data[data[:, 6] == 'r', 6] = 7
data[data[:, 6] == 'q', 6] = 8
data[data[:, 6] == 'w', 6] = 9
data[data[:, 6] == 'x', 6] = 10
data[data[:, 6] == 'e', 6] = 11
data[data[:, 6] == 'aa', 6] = 12
data[data[:, 6] == 'ff', 6] = 13
data[data[:, 6] == '?', 6] = -1

data[data[:, 7] == 'v', 7] = 0
data[data[:, 7] == 'h', 7] = 1
data[data[:, 7] == 'bb', 7] = 2
data[data[:, 7] == 'j', 7] = 3
data[data[:, 7] == 'n', 7] = 4
data[data[:, 7] == 'z', 7] = 5
data[data[:, 7] == 'dd', 7] = 6
data[data[:, 7] == 'ff', 7] = 7
data[data[:, 7] == 'o', 7] = 8
data[data[:, 7] == '?', 7] = -1

data[data[:, 8] == '?', 8] = -1

data[data[:, 9] == 't', 9] = 0
data[data[:, 9] == 'f', 9] = 1
data[data[:, 9] == '?', 9] = -1

data[data[:, 10] == 't', 10] = 0
data[data[:, 10] == 'f', 10] = 1
data[data[:, 10] == '?', 10] = -1

data[data[:, 11] == '?', 11] = -1

data[data[:, 12] == 't', 12] = 0
data[data[:, 12] == 'f', 12] = 1
data[data[:, 12] == '?', 12] = -1

data[data[:, 13] == 'g', 13] = 0
data[data[:, 13] == 'p', 13] = 1
data[data[:, 13] == 's', 13] = 2
data[data[:, 13] == '?', 13] = -1

data[data[:, 14] == '?', 14] = -1
data[data[:, 15] == '?', 15] = -1

y, X = data[:, 0].astype(int), data[:, 1:].astype(float)
y = np.asarray([[1, 0] if (_ == 0) else [0, 1] for _ in y])

# ====================================================
learning_rate = [0.001, 0.005, 0.01, 0.05]
learning_rule = ['sgd', 'adagrad']
hidden_units = [8, 16, 32, 64, 128, 256, 512]
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
output['score'] = [['learning_rate', 'learning_rule', 'hidden_units', 'n_iters', '#', 'acc', 'fpr', 'tpr', 'auc_manual', 'auc', 'countingtime']]


for lrt in learning_rate:
    for lr in learning_rule:
        for hu in hidden_units:
            for ni in n_iters:
                for ii in range(1):
                    print('learning_rate = ', lrt, 'learning_rule = ', lr, 'hidden_units = ', hu, 'n_iters = ', ni,
                          '## = ', ii)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

                    x = tf.placeholder(tf.float32, [None, X_train.shape[1]])

                    _initWeight = initWeight
                    W1, b1 = _initWeight([X_train.shape[1], hu]), _initWeight([hu])
                    W2, b2 = _initWeight([hu, y_train.shape[1]]), _initWeight([y_train.shape[1]])

                    a = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
                    yy = tf.nn.softmax(tf.matmul(a, W2) + b2)
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


                    fpr, tpr, threshold = getScore('ROC', y_test, y_hat)
                    score_auc = getScore('AUC', y_test, y_hat)

                    tmp = np.append(X_test, np.reshape(y_test, (1, y_test.shape[0])).T, axis=1)
                    tmp = np.append(tmp, np.reshape(y_hat, (1, y_hat.shape[0])).T, axis=1)
                    tmp = np.append(tmp, y_score, axis=1)
                    # tmp = np.append(tmp, np.asarray(ys), axis=1)

                    output['data'] = [['X_' + str(i) for i in range(1, X_test.shape[1] + 1)] +
                                      ['y_label', 'y_hat', 'y_score', 'y_score']] + \
                                     tmp.tolist()

                    output['score'].append([lrt, lr, hu, ni, ii, acc, fpr[1], tpr[1], metrics.auc(fpr, tpr), score_auc,
                                            (endTime - startTime).seconds + (
                                            endTime - startTime).microseconds / 10 ** 6])

saveXLSX(output, 'tensor_mlp_2classes_jcs.xlsx')