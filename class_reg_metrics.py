import numpy as np


def accuracy(actual,pred):
    return np.sum(pred==actual )/len(actual)


def confusion_matrix1(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1
    return cm

def mae(actual, pred):
    sum_err = 0
    for i in range(len(actual)):
        sum_err += abs(pred[i] - actual[i])
    return sum_err/float(len(actual))


def rmse(actual , pred):
    sq_pred = 0
    for i in range(len(actual)):
        sq_pred += (actual[i] - pred[i]) **2
    return np.sqrt(sq_pred/float(len(actual)))