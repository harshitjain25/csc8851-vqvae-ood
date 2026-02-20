import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

def fpr_at_95_tpr(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    return float(fpr[idx])

def compute_all(y_true, scores):
    return {
        "AUROC": float(roc_auc_score(y_true, scores)),
        "AUPR": float(average_precision_score(y_true, scores)),
        "FPR@95TPR": float(fpr_at_95_tpr(y_true, scores)),
    }