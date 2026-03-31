"""
evaluate_and_plot.py - ROC-AUC evaluation and plotting for Part 2.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score

from config import OUTPUT_DIR


def plot_roc_curve(y_true, y_scores, save_path: str = None):
    """
    Compute and save the ROC-AUC curve with additional metrics.

    Args:
        y_true:    Ground truth labels (0 = Real, 1 = Fake).
        y_scores:  Predicted probabilities from the classifier.
        save_path: Path to save the PNG. Defaults to outputs/roc_curve.png.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "roc_curve.png")

    if len(set(y_true)) < 2:
        print("[Eval] WARNING: Only one class in y_true — ROC-AUC undefined. Skipping.")
        return

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Best threshold via Youden's J statistic
    opt_idx   = int(np.argmax(tpr - fpr))
    opt_thr   = thresholds[opt_idx]
    y_pred    = [1 if s >= opt_thr else 0 for s in y_scores]
    acc       = accuracy_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random baseline")
    plt.scatter(fpr[opt_idx], tpr[opt_idx], color="red", s=100, zorder=5,
                label=f"Best threshold = {opt_thr:.3f}")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve — DIRE Fake Image Detection (SSD-1B / MPS)")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n[Eval] ROC curve saved: {save_path}")
    print(f"[Eval] AUC:       {roc_auc:.4f}")
    print(f"[Eval] Accuracy:  {acc:.4f}")
    print(f"[Eval] F1 Score:  {f1:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Evaluation with mock data")
    print("=" * 60)
    mock_true   = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
    mock_scores = [0.1, 0.9, 0.8, 0.2, 0.85, 0.3, 0.4, 0.95, 0.7, 0.15]
    plot_roc_curve(mock_true, mock_scores)
    print("Mock test PASSED.")
