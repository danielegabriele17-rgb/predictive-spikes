"""Cross-validation, out-of-fold evaluation, and leakage prevention utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


@dataclass
class CVResult:
    """Container for cross-validation metrics and out-of-fold predictions."""

    experiment_id: str
    cv_mean_acc: float
    cv_std_acc: float
    fold_val_accs: List[float]
    fold_train_accs: List[float]
    oof_Accuracy: float
    oof_roc_auc: float
    oof_log_loss: float
    n_features: int
    runtime_sec: float
    oof_probs: np.ndarray = field(repr=False)
    oof_preds: np.ndarray = field(repr=False)
    feature_names: List[str] = field(default_factory=list, repr=False)
    models: List[Any] = field(default_factory=list, repr=False)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary dictionary for experiment logging."""
        return {
            "experiment_id": self.experiment_id,
            "cv_mean_acc": round(float(self.cv_mean_acc), 6),
            "cv_std_acc": round(float(self.cv_std_acc), 6),
            "oof_accuracy": round(float(self.oof_Accuracy), 6),
            "oof_roc_auc": round(float(self.oof_roc_auc), 6),
            "oof_log_loss": round(float(self.oof_log_loss), 6),
            "train_mean_acc": round(float(np.mean(self.fold_train_accs)), 6),
            "train_val_gap": round(float(np.mean(self.fold_train_accs) - self.cv_mean_acc), 6),
            "fold_val_accs": [round(float(x), 6) for x in self.fold_val_accs],
            "fold_train_accs": [round(float(x), 6) for x in self.fold_train_accs],
            "n_features": int(self.n_features),
            "runtime_sec": round(float(self.runtime_sec), 3),
        }


def compute_battle_groups(battles: List[Dict[str, Any]]) -> np.ndarray:
    """Compute a deterministic hash per battle (excluding battle_id and player_won)
    so exact duplicate battles are always assigned to the same CV fold.
    """
    groups: List[str] = []
    for b in battles:
        b_clean = {k: v for k, v in b.items() if k not in ("battle_id", "player_won")}
        raw = json.dumps(b_clean, sort_keys=True)
        groups.append(hashlib.md5(raw.encode("utf-8")).hexdigest())
    return np.array(groups)


def get_cv_splits(
    battles: List[Dict[str, Any]],
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    group_duplicates: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate reproducible stratified K-fold splits."""
    if group_duplicates:
        groups = compute_battle_groups(battles)
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(sgkf.split(np.zeros(len(y)), y, groups))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def optimize_threshold_oof(
    y_true: np.ndarray,
    oof_probs: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[float, float, float]:
    """Evaluate default 0.5 decision threshold vs nested out-of-fold threshold tuning.

    Uses vectorized NumPy broadcasting `(probs[:, None] >= thresholds[None, :])`
    instead of Python loops to evaluate all 201 candidate thresholds simultaneously.
    Returns `(best_global_threshold, nested_cv_acc_with_tuned_threshold, default_05_acc)`.
    """
    default_acc = float(accuracy_score(y_true, (oof_probs >= 0.5).astype(int)))
    thresholds = np.linspace(0.40, 0.60, 201)

    nested_preds = np.zeros_like(y_true)
    for tr_idx, va_idx in splits:
        tr_probs = oof_probs[tr_idx]
        tr_y = y_true[tr_idx]
        # Vectorized accuracy evaluation across all 201 thresholds on the training fold
        accs = np.mean((tr_probs[:, None] >= thresholds[None, :]) == tr_y[:, None], axis=0)
        best_t = float(thresholds[int(np.argmax(accs))])
        nested_preds[va_idx] = (oof_probs[va_idx] >= best_t).astype(int)

    nested_acc = float(accuracy_score(y_true, nested_preds))
    global_accs = np.mean((oof_probs[:, None] >= thresholds[None, :]) == y_true[:, None], axis=0)
    best_global_t = float(thresholds[int(np.argmax(global_accs))])
    return best_global_t, nested_acc, default_acc
