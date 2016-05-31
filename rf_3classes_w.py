from dataset.store import loadCSV
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from libs.roc import getScore
from store import saveXLSX
from sklearn.metrics import confusion_matrix

folder = 'dataset/3 classes/wine'
file = 'wine.data.csv'

# ====================================================

data = np.asarray(loadCSV(folder, file))
y, X = data[:, 0].astype(int), data[:, 1:].astype(float)

y[y == 1] = 0
y[y == 2] = 1
y[y == 3] = 2

# ====================================================

output = {}
output['score'] = [['#', '##', 'n_estimators', 'acc', 'u', 'VUS_1', 'VUS_2', 'TP1', 'F12', 'F13', 'F21', 'TP2', 'F23', 'F31', 'F32', 'TP3', 'RunningTime']]

n_estimators = [8, 16, 32, 64, 128, 256, 512, 1024]

for jj in range(len(n_estimators)):
    for ii in range(10):
        print('j = ', jj, 'i = ', ii)
        # Train & Test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # ====================================================

        clf = RandomForestClassifier(n_estimators[jj])

        from datetime import datetime

        startTime = datetime.now()
        clf.fit(X_train, y_train)
        endTime = datetime.now()

        y_score = clf.predict_proba(X_test)
        y_hat = clf.predict(X_test)
        ys = [y_s[y_h-1] for y_s, y_h in zip(y_score, y_hat)]

        tmp = np.append(X_test, np.reshape(y_test, (1,y_test.shape[0])).T, axis=1)
        tmp = np.append(tmp, np.reshape(y_hat, (1,y_hat.shape[0])).T, axis=1)
        tmp = np.append(tmp, y_score, axis=1)
        tmp = np.append(tmp, np.asarray([ys]).T, axis=1)

        output['data'] = [['X_' + str(i) for i in range(1, X_test.shape[1] + 1)] +
                                ['y_label', 'y_hat', 'y_score_0', 'y_score_1', 'y_score_2', 'ys']] + \
                         tmp.tolist()

        acc = accuracy_score(y_hat, y_test)
        confMatrix = confusion_matrix(y_test, y_hat).tolist()

        score_u = getScore('U', y_test, y_score)
        score_VUS_1 = getScore('VUS_1', y_test, y_hat)
        score_VUS_2 = getScore('VUS_2', y_test, y_score)

        __ = (endTime - startTime)
        output['score'].append([jj, ii, n_estimators[jj], acc, score_u, score_VUS_1, score_VUS_2]
                               + confMatrix[0] + confMatrix[1] + confMatrix[2] + [__.seconds + __.microseconds/10**6])

saveXLSX(output, 'results/rf_3classes_w.xlsx')

None
