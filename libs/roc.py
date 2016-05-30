import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import mannwhitneyu
import scipy

def getScore(type='', y_true = [], y_score = []):

    if (type == 'ROC'):
        return roc_curve(y_true, y_score)

    if (type == 'AUC'):
        return roc_auc_score(y_true, y_score)

    elif (type == 'U0'):
        l0 = np.asarray([j[0] for (i, j) in zip(y_true, y_score.tolist()) if (i == 1)])
        l1 = np.asarray([j[1] for (i, j) in zip(y_true, y_score.tolist()) if (i == 2)])
        l2 = np.asarray([j[2] for (i, j) in zip(y_true, y_score.tolist()) if (i == 3)])
        return calU0((l0, l1, l2))

    # elif (type == 'U'):
    #     l0 = [j.index(max(j)) for (i, j) in zip(y_true, y_score.tolist()) if (i == 1)]
    #     l1 = [j.index(max(j))for (i, j) in zip(y_true, y_score.tolist()) if (i == 2)]
    #     l2 = [j.index(max(j)) for (i, j) in zip(y_true, y_score.tolist()) if (i == 3)]
    #     return calU((l0, l1, l2))

    # elif (type == 'U_S'):
    #     # return mannwhitneyu(y_true, y_score)
    #     return calU_S(y_true, y_score)
    #
    # elif (type == 'U_MannWhitneyu'):
    #     return mannwhitneyu(y_true, y_score)

    elif (type == 'VUS_1'):
        return calVUS_1(y_true, y_score)

    elif (type == 'VUS_2'):
        l0 = [j.index(max(j)) for (i, j) in zip(y_true, y_score.tolist()) if (i == 1)]
        l1 = [j.index(max(j)) for (i, j) in zip(y_true, y_score.tolist()) if (i == 2)]
        l2 = [j.index(max(j)) for (i, j) in zip(y_true, y_score.tolist()) if (i == 3)]
        return calVUS_2((l0, l1, l2))

    return None


# =================================================

def calU0(y_score):
    from scipy.stats import norm

    mu1, mu2, mu3 = np.mean(y_score[0]), np.mean(y_score[1]), np.mean(y_score[2])
    sigma1, sigma2, sigma3 = np.std(y_score[0]), np.std(y_score[1]), np.std(y_score[2])
    a, b, c, d = sigma2/sigma1, (mu1 - mu2)/sigma1, sigma2/sigma3, (mu3 - mu2)/sigma3

    bins, minS, maxS = 5000, -3, 3

    rg = np.arange(minS, maxS, (maxS - minS)/bins)

    import scipy.integrate as spi

    return spi.quad(lambda x: norm.pdf(a * x - b) * norm.pdf(-c * x + d) * norm.pdf(x), -5, 5)[0]

    # return integrate(norm.pdf(a * rg - b) * norm.pdf(-c * rg + d) * norm.pdf(rg) * ((maxS - minS)/bins))[0]

# =================================================
# 1. Mann-Whitney U Statistic
def calU(y_score):
    count = sum([1 for i in y_score[0] for j in y_score[1] for k in y_score[2] if (i < j) and (j < k)])
    return count / (len(y_score[0]) * len(y_score[1]) * len(y_score[2]))

# 1'
def calU_S(x, y):
    # u, prob = scipy.stats.mannwhitneyu(x, y)
    #
    # m_u = len(x) * len(y) / 2
    # sigma_u = np.sqrt(len(x) * len(y) * (len(x) + len(y) + 1) / 12)
    # z = (u - m_u) / sigma_u
    #
    # pval = 2 * scipy.stats.norm.cdf(z)
    #
    # return pval
    return 0

# =================================================
# 2. Approach based on the confusion matrix

def calPVUS(cfm, i, j, k):
    return (cfm[i][i] / (cfm[i][i] + cfm[i][j] + cfm[i][k])) * cfm[j][j] / (cfm[j][j] + cfm[j][k])

def calVUS_1(y_true, y_predicted):
    from sklearn.metrics import confusion_matrix
    import itertools

    confMatrix = confusion_matrix(y_true, y_predicted)

    tmp = [calPVUS(confMatrix, _[0], _[1], _[2]) for _ in list(itertools.permutations(range(3)))]
    return sum(tmp) / 6



# =================================================
# 3. Approach based on emperical distribution functions
#
# def calVUS_2(y_true, y_score):
#     return 0

def calPDF(data):
    return np.asarray([x / sum(data) for x in data])

def calCDF(data):
    return np.cumsum(calPDF(data))

def calVUS_2(data):
    # database = { 1 : [S_i], 2: [S_j], 3: [S_k] } for i, j, k in N

    bins = 100
    minS = min([min(data[0]), min(data[1]), min(data[2])])
    maxS = max([max(data[0]), max(data[1]), max(data[2])])

    count_S1, rangeS = np.histogram(np.asarray(data[0]), bins=bins, range=(minS, maxS))
    count_S2, tmp = np.histogram(np.asarray(data[1]), bins=bins, range=(minS, maxS)) #[0]
    count_S3, tmp = np.histogram(np.asarray(data[2]), bins=bins, range=(minS, maxS))  # [0]

    cdf1 = calCDF(count_S1)
    pdf2 = calPDF(count_S2)
    cdf3 = calCDF(count_S3)

    return sum(cdf1 * (1 - cdf3) * pdf2)
