from sklearn.metrics import confusion_matrix, roc_auc_score, auc, roc_curve
import numpy as np


def range_auc(true_y, predicted_score, FPR_range=None):
    # sorted_true_y, index = torch.sort(true_y, descending=True)
    # sorted_score = predict_score[index]
    # FP = TP = 0
    # FP_prev = TP_prev = 0
    # area = 0
    # f_prev = np.Inf
    # for i < len(sorted_true_y):
    #     if sorted_score[i] != f_prev:
    """Compute Area Under the Curve (AUC) using the trapezoidal rule.
    This is a general function, given points on a curve.  For computing the
    area under the ROC-curve, see :func:`roc_auc_score`.  For an alternative
    way to summarize a precision-recall curve, see
    :func:`average_precision_score`.
    Parameters
    ----------
    x : ndarray of shape (n,)
                                    x coordinates. These must be either monotonic increasing or monotonic
                                    decreasing.
    y : ndarray of shape, (n,)
                                    y coordinates.
    Returns
    -------
    auc : float
    See Also
    --------
    roc_auc_score : Compute the area under the ROC curve.
    average_precision_score : Compute average precision from prediction scores.
    precision_recall_curve : Compute precision-recall pairs for different
                                    probability thresholds.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([1, 1, 2, 2])
    >>> pred = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
    >>> metrics.auc(fpr, tpr)
    0.75
    """
    if FPR_range is not None:
        range1 = FPR_range[0]
        range2 = FPR_range[1]
        if (range1 >= range2):
            raise Exception('FPR range2 must be greater than range1')

    fpr, tpr, thresholds = roc_curve(true_y, predicted_score, pos_label=1)
    x = fpr
    y = tpr

    y1 = np.append(y, np.interp(range1, x, y))
    y = np.append(y1, np.interp(range2, x, y))
    x = np.append(x, range1)
    x = np.append(x, range2)

    x = np.sort(x)
    y = np.sort(y)

    range1_idx = np.where(x == range1)[-1][-1]
    range2_idx = np.where(x == range2)[-1][-1]

    trim_x = x[range1_idx:range2_idx + 1]
    trim_y = y[range1_idx:range2_idx + 1]

    area = auc(trim_x, trim_y)
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
