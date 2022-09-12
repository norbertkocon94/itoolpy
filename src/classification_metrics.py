# Libraries
import numpy as np

from sklearn.metrics import accuracy_score, precision_score


class Metrics:
    def __init__(self):
        pass

    def accuracy(self, y_true, y_pred, round=False):
        if round:
            return round(accuracy_score(y_true, y_pred), 2)
        else:
            return accuracy_score(y_true, y_pred)
    
    def precision(self, y_true, y_pred, round=False):
        if round:
            return precision_score(y_true, y_pred, round=False)
        else:
            return round(precision_score(y_true, y_pred))