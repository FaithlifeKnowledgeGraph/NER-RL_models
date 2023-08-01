from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve, precision_recall_curve

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class ClassificationMetrics:
    def __init__(self, y_true, y_pred, threshold=0.5):
        self.y_true = y_true.numpy()
        self.y_pred = y_pred.detach().numpy()
        self.y_pred_label = (self.y_pred > threshold).astype(float)

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
    
    def plot_roc_curve(self):
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % self.auc_roc())
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
        plt.legend(loc="lower right")
        plt.show()

    def plot_precision_recall_curve(self):
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_pred)
        plt.figure()
        plt.plot(recall, precision, label='PR curve (area = %0.2f)' % self.auc_roc())
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.legend(loc="lower right")
        plt.show()
    
    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_true, self.y_pred_label)
        plt.figure(figsize=(10,7))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.show()