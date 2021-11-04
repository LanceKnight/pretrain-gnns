from sklearn.metrics import confusion_matrix, roc_auc_score, auc, roc_curve
import numpy as np


def calculate_logAUC(true_y, predicted_score, FPR_range=None):
    '''
    :param true_y:input as numpy
    :param predicted_score: input as numpy
    :param FPR_range: input in union format. For example, to input range between 0.001 and 0.1, enter: (0.001, 0.1)
    :return:
    '''
    if FPR_range is not None:
        range1 = np.log10(FPR_range[0])
        range2 = np.log10(FPR_range[1])
        if (range1 >= range2):
            raise Exception('FPR range2 must be greater than range1')
    # print(f'true_y:{true_y}, predicted_score:{predicted_score}')
    fpr, tpr, thresholds = roc_curve(true_y, predicted_score, pos_label=1)
    x = fpr
    y = tpr
    x = np.log10(x)

    y1 = np.append(y, np.interp(range1, x, y))
    y = np.append(y1, np.interp(range2, x, y))
    x = np.append(x, range1)
    x = np.append(x, range2)

    x = np.sort(x)
    # print(f'x:{x}')
    y = np.sort(y)
    # print(f'y:{y}')

    range1_idx = np.where(x == range1)[-1][-1]
    range2_idx = np.where(x == range2)[-1][-1]

    trim_x = x[range1_idx:range2_idx + 1]
    trim_y = y[range1_idx:range2_idx + 1]

    area = auc(trim_x, trim_y)/2
    return area

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
