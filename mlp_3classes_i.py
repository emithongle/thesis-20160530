from dataset.store import loadCSV
import numpy as np
from sknn.mlp import Classifier, Layer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from libs.roc import getScore
from store import saveXLSX
import datetime

folder = 'dataset/3 classes/iris'
file = 'iris.data'

# ====================================================

data = np.asarray(loadCSV(folder, file))
# y = []
# X = []
# for _ in data:
#     y.append(_[4])
#     X.append(_[:4])

y, X = data[:, 4], data[:, :4].astype(float)
# y, X = np.asarray(y), np.asarray(X)

y[y == 'Iris-setosa'] = 0
y[y == 'Iris-versicolor'] = 1
y[y == 'Iris-virginica'] = 2
y = y.astype(int)

# ====================================================

learning_rate = [0.001, 0.005, 0.01, 0.05]
learning_rule = ['sgd', 'adagrad']
hidden_units = [8, 16, 32, 64, 128, 256, 512]
n_iters =[16, 32, 64, 128, 256, 512, 1024]

output = {}
output['score'] = [['learning_rate', 'learning_rule', 'hidden_units', 'n_iters', '#', 'acc', 'u', 'VUS_1', 'VUS_2', 'TP1', 'F12', 'F13', 'F21', 'TP2', 'F23', 'F31', 'F32', 'TP3' 'countingtime']]

for lrt in learning_rate:
    for lr in learning_rule:
        for hu in hidden_units:
            for ni in n_iters:
                for ii in range(10):
                    print('learning_rate = ', lrt, 'learning_rule = ', lr, 'hidden_units = ', hu, 'n_iters = ', ni, '## = ', ii)
                    # Train & Test
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

                    # ====================================================

                    clf = Classifier(
                        layers=[Layer('Sigmoid', units=hu), Layer('Softmax', units=3)],
                        learning_rule=lr,
                        learning_rate=lrt,
                        n_iter=ni
                    )

                    startTime = datetime.datetime.now()

                    clf.fit(X_train, y_train)

                    endTime = datetime.datetime.now()

                    y_score = clf.predict_proba(X_test)
                    y_hat = clf.predict(X_test)
                    ys = [y_s[y_h-1] for y_s, y_h in zip(y_score, y_hat)]

                    tmp = np.append(X_test, np.reshape(y_test, (1,y_test.shape[0])).T, axis=1)
                    tmp = np.append(tmp, np.reshape(y_hat, (1,y_hat.shape[0])).T, axis=1)
                    tmp = np.append(tmp, y_score, axis=1)
                    tmp = np.append(tmp, np.asarray(ys), axis=1)

                    output['data'] = [['X_' + str(i) for i in range(1, X_test.shape[1] + 1)] +
                                            ['y_label', 'y_hat', 'y_score_1', 'y_score_2', 'y_score_3', 'ys']] + \
                                     tmp.tolist()

                    acc = accuracy_score(y_hat, y_test)

                    confMatrix = confusion_matrix(y_test, y_hat).tolist()

                    # fpr, tpr, threshold = getScore('ROC', y_test, y_hat)

                    # score_auc = getScore('AUC', y_test, y_hat)
                    score_u = getScore('U', y_test, y_score)
                    # score_u_s = getScore('U_S', y_test, ys)
                    # score_u_m = getScore('U_MannWhitneyu', y_test, np.asarray(ys).T[0])
                    score_VUS_1 = getScore('VUS_1', y_test, y_hat)
                    score_VUS_2 = getScore('VUS_2', y_test, y_score)

                    output['score'].append(
                        [lrt, lr, hu, ni, ii, acc, score_u, score_VUS_1, score_VUS_2]
                            + confMatrix[0] + confMatrix[1] + confMatrix[2] +
                         [(endTime - startTime).seconds + (endTime - startTime).microseconds / 10 ** 6])

saveXLSX(output, 'results/mlp_3classes_i.xlsx')
