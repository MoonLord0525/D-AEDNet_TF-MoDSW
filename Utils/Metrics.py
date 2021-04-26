import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

'''
The order return is Accuracy, Precision, Recall, F1_Score and Specificity
'''


def EvaluationMetrics(y_pred, y_true):
    if not isinstance(y_pred, np.int32):
        y_pred = y_pred.astype(np.int32)
    if not isinstance(y_true, np.int32):
        y_true = y_true.astype(np.int32)
    Metrics = np.zeros(shape=(5, y_pred.shape[0]), dtype=np.float32)
    for i in range(y_pred.shape[0]):
        ConfusionMatrix = confusion_matrix(y_true=y_true[i], y_pred=y_pred[i][0][0])
        Metrics[0][i] = accuracy_score(y_true=y_true[i], y_pred=y_pred[i][0][0])
        Metrics[1][i] = precision_score(y_true=y_true[i], y_pred=y_pred[i][0][0])
        Metrics[2][i] = recall_score(y_true=y_true[i], y_pred=y_pred[i][0][0])
        Metrics[3][i] = f1_score(y_true=y_true[i], y_pred=y_pred[i][0][0])
        Metrics[4][i] = ConfusionMatrix[0][0] / (ConfusionMatrix[0][0] + ConfusionMatrix[0][1])
    return np.around(Metrics.mean(axis=1), decimals=3)