"""Metric computation and decision-threshold tuning.

XGBoost outputs a fraud probability; converting that into a block/allow
decision requires picking an operating threshold. The default 0.5 cutoff is
rarely right for a problem this imbalanced, so the threshold is tuned
explicitly on a held-out validation split rather than hard-coded.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def tune_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Sweep candidate thresholds from the precision-recall curve and return
    the one that maximises F1, alongside a higher-recall alternative
    (useful when the business cost of a missed fraud outweighs the cost of a
    false positive).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns one more point than thresholds; align them.
    precisions, recalls = precisions[:-1], recalls[:-1]
    f1_scores = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(precisions),
        where=(precisions + recalls) > 0,
    )
    best_idx = int(np.argmax(f1_scores))

    high_recall_candidates = np.where(recalls >= 0.80)[0]
    high_recall_idx = (
        high_recall_candidates[np.argmax(precisions[high_recall_candidates])]
        if len(high_recall_candidates) > 0
        else best_idx
    )

    return {
        "f1_optimal_threshold": float(thresholds[best_idx]),
        "f1_optimal_precision": float(precisions[best_idx]),
        "f1_optimal_recall": float(recalls[best_idx]),
        "high_recall_threshold": float(thresholds[high_recall_idx]),
        "high_recall_precision": float(precisions[high_recall_idx]),
        "high_recall_recall": float(recalls[high_recall_idx]),
    }


def evaluate_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }
