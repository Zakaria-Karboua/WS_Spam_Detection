"""Metrics: accuracy, precision, recall, F1, confusion matrix, false-positive rate."""
import numpy as np


def _div(a, b):
    return float(a) / float(b) if b else 0.0


def compute_metrics(y_true, y_pred):
    """Spam (1) is the positive class. A false positive is a legitimate message
    classified as spam, which is the costly error in practice."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    return {
        "accuracy": _div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": _div(2 * precision * recall, precision + recall),
        "fpr": _div(fp, fp + tn),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
