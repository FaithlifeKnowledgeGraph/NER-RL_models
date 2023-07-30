from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix


class ClassificationMetrics:
    def __init__(self, y_true, y_pred, threshold=0.5):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_label = (y_pred > threshold).float()

    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred_label)

    def precision(self):
        return precision_score(self.y_true, self.y_pred_label)

    def recall(self):
        return recall_score(self.y_true, self.y_pred_label)

    def f1(self):
        return f1_score(self.y_true, self.y_pred_label)

    def auc_roc(self):
        return roc_auc_score(self.y_true, self.y_pred_label)
    
    def calc_confusion_matrix(self):
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred_label).ravel()
        return tn, fp, fn, tp
