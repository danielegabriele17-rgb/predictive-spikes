"""XGBoost model training, native C++ Booster feature importance, and 5-fold CV inference."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

from src.validation import CVResult


# Default histogram-based Gradient Boosted Decision Tree hyperparameters
DEFAULT_XGB_PARAMS: Dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": 4,
}


def build_xgb_model(
    params: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    early_stopping_rounds: Optional[int] = None,
) -> XGBClassifier:
    """Instantiate an `XGBClassifier` with competition-compliant defaults."""
    cfg = dict(DEFAULT_XGB_PARAMS)
    if params:
        cfg.update(params)
    cfg["random_state"] = seed
    if early_stopping_rounds is not None:
        cfg["early_stopping_rounds"] = early_stopping_rounds
    else:
        cfg.pop("early_stopping_rounds", None)
    return XGBClassifier(**cfg)


def run_xgb_cv(
    X_df: pd.DataFrame,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    xgb_params: Optional[Dict[str, Any]] = None,
    experiment_id: str = "xgb_experiment",
    early_stopping_rounds: Optional[int] = 50,
    fold_feature_fn: Optional[
        Callable[[np.ndarray, np.ndarray], Tuple[pd.DataFrame, pd.DataFrame]]
    ] = None,
    X_test_df: Optional[pd.DataFrame] = None,
    test_fold_feature_fn: Optional[
        Callable[[np.ndarray, np.ndarray], pd.DataFrame]
    ] = None,
    threshold: float = 0.5,
    seed: int = 42,
) -> Tuple[CVResult, Optional[np.ndarray]]:
    """Run stratified cross-validation with XGBoost and optional within-fold feature extraction.

    Any fold-dependent statistics (e.g., `WithinFoldTargetEncoder`) are appended strictly inside
    each fold via `fold_feature_fn(tr_idx, va_idx)` so zero target leakage occurs.
    """
    t0 = time.perf_counter()
    n_samples = len(y)
    oof_probs = np.zeros(n_samples, dtype=float)
    test_probs_folds: List[np.ndarray] = []

    fold_val_accs: List[float] = []
    fold_train_accs: List[float] = []
    models: List[XGBClassifier] = []
    feature_names: List[str] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(splits):
        X_tr = X_df.iloc[tr_idx].copy()
        X_va = X_df.iloc[va_idx].copy()
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        # Append strictly within-fold encoded features if provided
        if fold_feature_fn is not None:
            extra_tr, extra_va = fold_feature_fn(tr_idx, va_idx)
            X_tr = pd.concat([X_tr.reset_index(drop=True), extra_tr.reset_index(drop=True)], axis=1)
            X_va = pd.concat([X_va.reset_index(drop=True), extra_va.reset_index(drop=True)], axis=1)

        feature_names = list(X_tr.columns)

        model = build_xgb_model(
            params=xgb_params,
            seed=seed + fold_idx,
            early_stopping_rounds=early_stopping_rounds,
        )

        if early_stopping_rounds is not None:
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )
        else:
            model.fit(X_tr, y_tr, verbose=False)

        tr_probs = model.predict_proba(X_tr)[:, 1]
        va_probs = model.predict_proba(X_va)[:, 1]
        oof_probs[va_idx] = va_probs

        tr_acc = float(accuracy_score(y_tr, (tr_probs >= threshold).astype(int)))
        va_acc = float(accuracy_score(y_va, (va_probs >= threshold).astype(int)))
        fold_train_accs.append(tr_acc)
        fold_val_accs.append(va_acc)
        models.append(model)

        if X_test_df is not None:
            X_te = X_test_df.copy()
            if test_fold_feature_fn is not None:
                extra_te = test_fold_feature_fn(tr_idx, va_idx)
                X_te = pd.concat([X_te.reset_index(drop=True), extra_te.reset_index(drop=True)], axis=1)
            X_te = X_te[feature_names]
            test_probs_folds.append(model.predict_proba(X_te)[:, 1])

    runtime = time.perf_counter() - t0
    oof_preds = (oof_probs >= threshold).astype(int)
    oof_acc = float(accuracy_score(y, oof_preds))
    oof_auc = float(roc_auc_score(y, oof_probs))
    oof_ll = float(log_loss(y, oof_probs))

    cv_res = CVResult(
        experiment_id=experiment_id,
        cv_mean_acc=float(np.mean(fold_val_accs)),
        cv_std_acc=float(np.std(fold_val_accs)),
        fold_val_accs=fold_val_accs,
        fold_train_accs=fold_train_accs,
        oof_Accuracy=oof_acc,
        oof_roc_auc=oof_auc,
        oof_log_loss=oof_ll,
        n_features=len(feature_names),
        runtime_sec=runtime,
        oof_probs=oof_probs,
        oof_preds=oof_preds,
        feature_names=feature_names,
        models=models,
    )

    avg_test_probs = np.mean(test_probs_folds, axis=0) if test_probs_folds else None
    return cv_res, avg_test_probs


def get_feature_importance_df(models: List[XGBClassifier], feature_names: List[str]) -> pd.DataFrame:
    """Extract multi-metric native XGBoost feature importances across CV fold Boosters.

    Queries `model.get_booster().get_score(importance_type=...)` for each fold model to compute:
    - `importance`: Normalized average gain (`feature_importances_`).
    - `gain`: Native average loss reduction per split (`get_score(importance_type="gain")`).
    - `total_gain`: Total loss reduction across all splits (`get_score(importance_type="total_gain")`).
    - `weight`: Number of times the feature is used to split a node (`get_score(importance_type="weight")`).
    - `cover`: Average number of battles covered by splits on the feature (`get_score(importance_type="cover")`).
    """
    n_feats = len(feature_names)
    if not models:
        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance": np.zeros(n_feats),
                "gain": np.zeros(n_feats),
                "total_gain": np.zeros(n_feats),
                "weight": np.zeros(n_feats),
                "cover": np.zeros(n_feats),
            }
        )

    norm_gain = np.mean([m.feature_importances_ for m in models], axis=0)
    metric_accumulator: Dict[str, np.ndarray] = {
        m_type: np.zeros(n_feats, dtype=float)
        for m_type in ("gain", "total_gain", "weight", "cover")
    }
    feat_idx_map = {fname: idx for idx, fname in enumerate(feature_names)}

    for m in models:
        booster = m.get_booster()
        for m_type, arr in metric_accumulator.items():
            scores = booster.get_score(importance_type=m_type)
            for k, val in scores.items():
                idx = feat_idx_map.get(k)
                if idx is None and k.startswith("f") and k[1:].isdigit():
                    f_i = int(k[1:])
                    if 0 <= f_i < n_feats:
                        idx = f_i
                if idx is not None:
                    arr[idx] += float(val)

    n_models = float(len(models))
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": norm_gain,
                "gain": metric_accumulator["gain"] / n_models,
                "total_gain": metric_accumulator["total_gain"] / n_models,
                "weight": metric_accumulator["weight"] / n_models,
                "cover": metric_accumulator["cover"] / n_models,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
