"""Hyperparameter optimization, feature selection, and ensemble utilities for XGBoost."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from src.model import DEFAULT_XGB_PARAMS, build_xgb_model, get_feature_importance_df, run_xgb_cv
from src.validation import CVResult


optuna.logging.set_verbosity(optuna.logging.WARNING)


def select_top_features(
    importance_df: pd.DataFrame,
    top_k: Optional[int] = None,
    min_importance: float = 0.0,
) -> List[str]:
    """Select top features based on cross-validated XGBoost feature importance."""
    filtered = importance_df[importance_df["importance"] > min_importance].copy()
    if top_k is not None:
        filtered = filtered.head(top_k)
    return list(filtered["feature"].values)


def prune_collinear_features(
    X_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    corr_threshold: float = 0.985,
) -> List[str]:
    """Remove nearly redundant features (|r| > corr_threshold), always keeping the one
    with higher cross-validated feature importance.
    """
    ordered_feats = list(importance_df["feature"].values)
    X_ord = X_df[ordered_feats]
    corr_matrix = X_ord.corr().abs().to_numpy()
    n = len(ordered_feats)
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if keep[j] and corr_matrix[i, j] >= corr_threshold:
                keep[j] = False

    return [ordered_feats[i] for i in range(n) if keep[i]]


def optimize_xgb_optuna(
    X_df: pd.DataFrame,
    y: np.ndarray,
    tuning_splits: List[Tuple[np.ndarray, np.ndarray]],
    n_trials: int = 30,
    seed: int = 42,
    timeout_sec: Optional[int] = 600,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Run systematic hyperparameter search for XGBoost using Optuna TPE.

    Uses `tuning_splits` (without validation-peeking early stopping) so hyperparameter
    selection does not leak validation early stopping into the final CV evaluation.
    """
    trials_records: List[Dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params: Dict[str, Any] = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_estimators": trial.suggest_int("n_estimators", 250, 850, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 35),
            "subsample": trial.suggest_float("subsample", 0.60, 0.92),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.35, 0.85),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.55, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.2, 20.0, log=True),
            "n_jobs": 4,
        }

        cv_res, _ = run_xgb_cv(
            X_df=X_df,
            y=y,
            splits=tuning_splits,
            xgb_params=params,
            experiment_id=f"optuna_trial_{trial.number}",
            early_stopping_rounds=None,
            seed=seed,
        )

        rec = dict(params)
        rec["trial_number"] = trial.number
        rec["cv_mean_acc"] = cv_res.cv_mean_acc
        rec["cv_std_acc"] = cv_res.cv_std_acc
        rec["oof_log_loss"] = cv_res.oof_log_loss
        rec["oof_roc_auc"] = cv_res.oof_roc_auc
        rec["train_mean_acc"] = float(np.mean(cv_res.fold_train_accs))
        trials_records.append(rec)

        # Maximize CV Accuracy with tiny tie-breaker on negative log_loss
        return cv_res.cv_mean_acc - 0.002 * cv_res.oof_log_loss

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec)

    trials_df = pd.DataFrame(trials_records).sort_values("cv_mean_acc", ascending=False).reset_index(drop=True)
    best_params = dict(DEFAULT_XGB_PARAMS)
    best_params.update(study.best_params)
    return best_params, trials_df


def evaluate_oof_ensemble(
    y: np.ndarray,
    oof_probs_list: Sequence[np.ndarray],
    weights: Optional[Sequence[float]] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate weighted probability average of out-of-fold predictions."""
    if weights is None:
        weights = [1.0 / len(oof_probs_list)] * len(oof_probs_list)
    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w)

    blend_probs = np.zeros_like(oof_probs_list[0], dtype=float)
    for p, weight in zip(oof_probs_list, w):
        blend_probs += weight * p

    preds = (blend_probs >= threshold).astype(int)
    return {
        "oof_accuracy": round(float(accuracy_score(y, preds)), 6),
        "oof_roc_auc": round(float(roc_auc_score(y, blend_probs)), 6),
        "oof_log_loss": round(float(log_loss(y, blend_probs)), 6),
    }
