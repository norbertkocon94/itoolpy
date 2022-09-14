# Libraries
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def accuracy(y_true, y_pred, round=False):
    # * Function that returns accuracy score

    if round:
        return round(accuracy_score(y_true, y_pred), 2)
    else:
        return accuracy_score(y_true, y_pred)

def precision(y_true, y_pred, round=False):
    # * Function that returns precision score

    if round:
        return round(precision_score(y_true, y_pred), 2)
    else:
        return precision_score(y_true, y_pred)
        
def recall(y_true, y_pred, round=False):
    # * Function that returns recall score

    if round:
        return round(recall_score(y_true, y_pred), 2)
    else:
        return recall_score(y_true, y_pred)

def f1(y_true, y_pred, round=False):
    # * Function that returns f1 score

    if round:
        return round(f1_score(y_true, y_pred), 2)
    else:
        return f1_score(y_true, y_pred)

def report(y_true, y_pred):
    # * Function that returns classification report
    # * with 4 metrics (Accuracy, Precision, Recall, F1)

    # All metrics saved variables
    accuracy =  accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall =    recall_score(y_true, y_pred)
    f1 =        f1_score(y_true, y_pred)

    # Create dataframe with merics
    report = {'Accuracy':accuracy, 'Precision':precision, 'Recall':recall, "F1":f1}
    report = pd.DataFrame.from_dict(report, orient='index', columns=['Value'])

    return report


