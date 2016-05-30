from dataset.store import loadCSV
import numpy as np
from sknn.mlp import Classifier, Layer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from libs.roc import getScore
from store import saveXLSX
from sklearn import metrics
from datetime import datetime

folder = 'dataset/2 classes/japanese_create_screening'
file = 'crx.csv'

# ====================================================

data = np.asarray(loadCSV(folder, file))

data[data[:, 0] == '-'] = 0
data[data[:, 0] == '+'] = 1

data[data[:, 1] == 'a'] = 0
data[data[:, 1] == 'b'] = 1

data[data[:, 4] == 'u'] = 0
data[data[:, 4] == 'y'] = 1
data[data[:, 4] == 'l'] = 2
data[data[:, 4] == 't'] = 3

data[data[:, 5] == 'g'] = 0
data[data[:, 5] == 'p'] = 1
data[data[:, 5] == 'gg'] = 2

data[data[:, 6] == 'c'] = 0
data[data[:, 6] == 'd'] = 1
data[data[:, 6] == 'cc'] = 2
data[data[:, 6] == 'i'] = 3
data[data[:, 6] == 'j'] = 4
data[data[:, 6] == 'k'] = 5
data[data[:, 6] == 'm'] = 6
data[data[:, 6] == 'r'] = 7
data[data[:, 5] == 'q'] = 8
data[data[:, 6] == 'w'] = 9
data[data[:, 6] == 'x'] = 10
data[data[:, 6] == 'e'] = 11
data[data[:, 6] == 'aa'] = 12
data[data[:, 6] == 'ff'] = 13

data[data[:, 7] == 'v'] = 0
data[data[:, 7] == 'h'] = 1
data[data[:, 7] == 'bb'] = 2
data[data[:, 7] == 'j'] = 3
data[data[:, 7] == 'n'] = 4
data[data[:, 7] == 'z'] = 5
data[data[:, 7] == 'dd'] = 6
data[data[:, 7] == 'ff'] = 7
data[data[:, 7] == 'o'] = 8

data[data[:, 9] == 't'] = 0
data[data[:, 9] == 'f'] = 1

data[data[:, 10] == 't'] = 0
data[data[:, 10] == 'f'] = 1

data[data[:, 12] == 't'] = 0
data[data[:, 12] == 'f'] = 1

data[data[:, 13] == 'g'] = 0
data[data[:, 13] == 'p'] = 1
data[data[:, 13] == 's'] = 2

y, X = data[:, 0].astype(int), data[:, 1:].astype(float)

# ====================================================
learning_rate = [0.01] # [0.001, 0.005, 0.01, 0.05]
learning_rule = ['sgd'] # ['sgd', 'adagrad']
hidden_units = [8] # [8, 16, 32, 64, 128, 256, 512]
n_iters = [16] # [16, 32, 64, 128, 256, 512, 1024]

output = {}

output['score'] = [['learning_rate', 'learning_rule', 'hidden_units', 'n_iters', '#', 'acc', 'fpr', 'tpr', 'auc_manual', 'auc', 'countingtime']]

for lrt in learning_rate:
    for lr in learning_rule:
        for hu in hidden_units:
            for ni in n_iters:
                for ii in range(1):
                    print('learning_rate = ', lrt, 'learning_rule = ', lr, 'hidden_units = ', hu, 'n_iters = ', ni, '## = ', ii)

                    # Train & Test
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

                    # ====================================================

                    clf = Classifier(
                        layers=[Layer('Sigmoid', units=hu), Layer('Softmax', units=2)],
                        learning_rule=lr,
                        learning_rate=lrt,
                        n_iter=ni
                    )

                    startTime = datetime.now()

                    clf.fit(X_train, y_train)

                    endTime = datetime.now()

                    y_score = clf.predict_proba(X_test)
                    y_hat = clf.predict(X_test)
                    ys = [y_s[y_h] for y_s, y_h in zip(y_score, y_hat)]

                    tmp = np.append(X_test, np.reshape(y_test, (1, y_test.shape[0])).T, axis=1)
                    tmp = np.append(tmp, np.reshape(y_hat, (1, y_hat.shape[0])).T, axis=1)
                    tmp = np.append(tmp, y_score, axis=1)
                    tmp = np.append(tmp, np.asarray(ys), axis=1)

                    output['data'] = [['X_' + str(i) for i in range(1, X_test.shape[1] + 1)] +
                                      ['y_label', 'y_hat', 'y_score', 'y_score', 'ys']] + \
                                     tmp.tolist()

                    acc = accuracy_score(y_hat, y_test)

                    fpr, tpr, threshold = getScore('ROC', y_test, y_hat)

                    score_auc = getScore('AUC', y_test, y_hat)

                    output['score'].append([lrt, lr, hu, ni, ii, acc, fpr[1], tpr[1], metrics.auc(fpr, tpr), score_auc, (endTime - startTime).seconds +  (endTime - startTime).microseconds/10**6])

saveXLSX(output, 'mlp_2classes_jcs.xlsx')