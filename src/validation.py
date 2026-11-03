"""Cross-validation and out-of-fold evaluation utilities for the Pokémon Battle pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold


@dataclass
class CVResult:
    """Container for 5-fold cross-validation metrics and out-of-fold predictions."""

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


def get_cv_splits(
    battles: List[Dict[str, Any]],
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    group_duplicates: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate reproducible stratified K-fold splits preserving class balance (`50% / 50%`)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))
