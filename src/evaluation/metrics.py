import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.utils.config import CLASSES

def compute_metrics(y_true, y_pred, average='weighted'):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average)
    rec = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1}

def compute_per_class_metrics(y_true, y_pred):
    report = {}
    cm = confusion_matrix(y_true, y_pred)
    for i, cls in enumerate(CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        report[cls] = {'precision': prec, 'recall': rec, 'f1_score': f1, 'support': tp + fn}
    return report

def evaluate_model(model, X_test, y_test):
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    global_metrics = compute_metrics(y_test, y_pred)
    per_class = compute_per_class_metrics(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return global_metrics, per_class, cm
