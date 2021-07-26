from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np


def log_auc(FPR_range=None):
    pass

def enrichment(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    p = np.sum(y_true == 1)
    n = np.sum(y_true == 0)
    print(f'tn:{tn} fp:{fp}, fn:{fn} tp:{tp} p:{p}, n:{n}')
    if(tp + fp) != 0:
        enr = (tp / (tp + fp)) / (p / (p + n))
    else:
        enr = np.NAN
    return enr


def ppv(y_true, y_pred):
    '''
    positve predictive value
    '''
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    if(tp + fp) != 0:
        ppv = (tp / (tp + fp))
    else:
        ppv = np.NAN
    return ppv


def roc_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


if __name__ == '__main__':
    y_true = np.array([1, 0, 0, 1, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0])
    out = enrichment(y_true, y_pred)
    print(out)
