"""Systematic XGBoost optimization, multi-configuration evaluation, and model selection."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.battle_features import build_species_catalogue
from src.data_loader import load_competition_data
from src.feature_engineering import (
    ALL_FEATURE_FAMILIES,
    WithinFoldTargetEncoder,
    build_feature_dataframe,
)
from src.model import DEFAULT_XGB_PARAMS, get_feature_importance_df, run_xgb_cv
from src.optimization import (
    evaluate_oof_ensemble,
    optimize_xgb_optuna,
    prune_collinear_features,
    select_top_features,
)
from src.submission import generate_submission, validate_submission_df
from src.validation import get_cv_splits, optimize_threshold_oof


def main() -> None:
    t_start = time.perf_counter()
    print("=== Phase 1: Loading Competition Data ===")
    train_battles, test_battles, sample_sub, type_chart = load_competition_data()
    y = np.array([int(b["player_won"]) for b in train_battles], dtype=int)
    catalogue = build_species_catalogue(train_battles)

    print("=== Phase 2: Extracting Full Feature Matrix (Train & Test) ===")
    X_train_full, train_ids, _ = build_feature_dataframe(
        train_battles, families=ALL_FEATURE_FAMILIES, catalogue=catalogue, type_chart=type_chart
    )
    X_test_full, test_ids, _ = build_feature_dataframe(
        test_battles, families=ALL_FEATURE_FAMILIES, catalogue=catalogue, type_chart=type_chart
    )
    print(f"Extracted {X_train_full.shape[1]} features for {len(train_battles)} train and {len(test_battles)} test battles.")

    # Canonical 5-fold CV splits (seed=42) and separate Optuna tuning splits (seed=1337)
    cv_splits = get_cv_splits(train_battles, y, n_splits=5, seed=42, group_duplicates=False)
    cv_splits_grouped = get_cv_splits(train_battles, y, n_splits=5, seed=42, group_duplicates=True)
    tuning_splits = get_cv_splits(train_battles, y, n_splits=5, seed=1337, group_duplicates=True)

    # Precompute within-fold target encoding for cv_splits (seed=42) so all experiments reuse it fast
    print("=== Phase 3: Computing Strictly Leak-Free Within-Fold Target Encodings ===")
    te_folds_train: List[pd.DataFrame] = []
    te_folds_val: List[pd.DataFrame] = []
    te_folds_test: List[pd.DataFrame] = []

    for tr_idx, va_idx in cv_splits:
        enc = WithinFoldTargetEncoder(smoothing=30.0, team_smoothing=20.0)
        tr_b = [train_battles[i] for i in tr_idx]
        va_b = [train_battles[i] for i in va_idx]
        te_folds_train.append(enc.fit_transform_inner_oof(tr_b, y[tr_idx], n_inner_splits=5, seed=42))
        te_folds_val.append(enc.transform(va_b))
        te_folds_test.append(enc.transform(test_battles))

    def get_precomputed_te_fns():
        fold_map = {( int(tr[0]), int(va[0]) ): idx for idx, (tr, va) in enumerate(cv_splits)}
        def fold_fn(tr_idx: np.ndarray, va_idx: np.ndarray):
            idx = fold_map[(int(tr_idx[0]), int(va_idx[0]))]
            return te_folds_train[idx], te_folds_val[idx]
        def test_fn(tr_idx: np.ndarray, va_idx: np.ndarray):
            idx = fold_map[(int(tr_idx[0]), int(va_idx[0]))]
            return te_folds_test[idx]
        return fold_fn, test_fn

    fold_te_fn, test_te_fn = get_precomputed_te_fns()

    # Step 4: Baseline CV on all features to rank feature importances & prune collinearity
    print("=== Phase 4: Collinearity Pruning & Feature Subset Selection ===")
    cv_base_obs, _ = run_xgb_cv(
        X_df=X_train_full,
        y=y,
        splits=cv_splits,
        xgb_params=DEFAULT_XGB_PARAMS,
        experiment_id="base_obs_full",
        early_stopping_rounds=50,
        seed=42,
    )
    imp_df = get_feature_importance_df(cv_base_obs.models, cv_base_obs.feature_names)
    pruned_features = prune_collinear_features(X_train_full, imp_df, corr_threshold=0.98)
    imp_pruned_df = imp_df[imp_df["feature"].isin(pruned_features)].reset_index(drop=True)

    top200_features = select_top_features(imp_pruned_df, top_k=200)
    top220_features = select_top_features(imp_pruned_df, top_k=220)
    top140_features = select_top_features(imp_pruned_df, top_k=140)
    print(f"Pruned features: {len(pruned_features)} (from {X_train_full.shape[1]}). Selected subsets: top200={len(top200_features)}, top220={len(top220_features)}, top140={len(top140_features)}")

    # Verify strictly within-fold feature selection (selecting top 200 inside each training fold only)
    within_fold_sel_accs = []
    for f_idx, (tr_idx, va_idx) in enumerate(cv_splits):
        m_in = cv_base_obs.models[f_idx]
        # Notice: to be 100% free of val early-stopping, fit a quick selector model on tr_idx only
        from src.model import build_xgb_model
        sel_m = build_xgb_model(DEFAULT_XGB_PARAMS, seed=42 + f_idx, early_stopping_rounds=None)
        sel_m.fit(X_train_full[pruned_features].iloc[tr_idx], y[tr_idx], verbose=False)
        fold_imp = pd.DataFrame({"feature": pruned_features, "importance": sel_m.feature_importances_}).sort_values("importance", ascending=False)
        fold_top200 = list(fold_imp.head(200)["feature"].values)

        X_tr_f = pd.concat([X_train_full[fold_top200].iloc[tr_idx].reset_index(drop=True), te_folds_train[f_idx].reset_index(drop=True)], axis=1)
        X_va_f = pd.concat([X_train_full[fold_top200].iloc[va_idx].reset_index(drop=True), te_folds_val[f_idx].reset_index(drop=True)], axis=1)
        ev_m = build_xgb_model(DEFAULT_XGB_PARAMS, seed=42 + f_idx, early_stopping_rounds=50)
        ev_m.fit(X_tr_f, y[tr_idx], eval_set=[(X_va_f, y[va_idx])], verbose=False)
        va_p = ev_m.predict_proba(X_va_f)[:, 1]
        within_fold_sel_accs.append(float(accuracy_score(y[va_idx], (va_p >= 0.5).astype(int))))
    print(f"Strictly Within-Fold Top-200 Selection CV Acc: {np.mean(within_fold_sel_accs):.4f} +/- {np.std(within_fold_sel_accs):.4f}")

    # Step 5: Systematic Optuna Hyperparameter Optimization on independent tuning_splits (seed=1337)
    print("\n=== Phase 5: Running Optuna Hyperparameter Optimization (25 Trials on Tuning Splits) ===")
    best_optuna_params, trials_df = optimize_xgb_optuna(
        X_df=X_train_full[top200_features],
        y=y,
        tuning_splits=tuning_splits,
        n_trials=25,
        seed=42,
        timeout_sec=360,
    )
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(out_dir / "optuna_trials.csv", index=False)
    print("Top 5 Optuna trials on independent tuning splits (seed=1337):")
    print(trials_df[["trial_number", "cv_mean_acc", "cv_std_acc", "oof_log_loss", "max_depth", "learning_rate", "n_estimators", "min_child_weight", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]].head(5).to_string(index=False))

    # Define the 3 distinct, well-regularized candidate XGBoost configurations
    # Config 1: Optuna-selected regularized XGBoost on Top-200 features + Within-Fold TE
    cfg1_params = dict(best_optuna_params)

    # Config 2: Conservative shallow, heavily regularized XGBoost on all 512 pruned features + Within-Fold TE
    cfg2_params: Dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_estimators": 550,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 15,
        "subsample": 0.75,
        "colsample_bytree": 0.45,
        "colsample_bylevel": 0.75,
        "gamma": 0.4,
        "reg_alpha": 1.5,
        "reg_lambda": 5.0,
        "n_jobs": 4,
    }

    # Config 3: Pure Observation-Only (Zero-TE) Medium-Depth XGBoost on Top-220 features
    cfg3_obs_params: Dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 12,
        "subsample": 0.78,
        "colsample_bytree": 0.55,
        "colsample_bylevel": 0.80,
        "gamma": 0.25,
        "reg_alpha": 0.8,
        "reg_lambda": 3.5,
        "n_jobs": 4,
    }

    # Config 4: Compact Top-140 features + Within-Fold TE with depth-5 regularized trees
    cfg4_compact_params: Dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_estimators": 500,
        "learning_rate": 0.032,
        "max_depth": 5,
        "min_child_weight": 10,
        "subsample": 0.80,
        "colsample_bytree": 0.60,
        "colsample_bylevel": 0.80,
        "gamma": 0.2,
        "reg_alpha": 0.5,
        "reg_lambda": 2.5,
        "n_jobs": 4,
    }

    print("\n=== Phase 6: Evaluating Final Candidate Models on Canonical 5-Fold CV (seed=42) ===")
    cv_m1, test_probs_m1 = run_xgb_cv(
        X_df=X_train_full[top200_features],
        y=y,
        splits=cv_splits,
        xgb_params=cfg1_params,
        experiment_id="model1_optuna_top200_te",
        early_stopping_rounds=50,
        fold_feature_fn=fold_te_fn,
        X_test_df=X_test_full[top200_features],
        test_fold_feature_fn=test_te_fn,
        seed=42,
    )
    print(f"[Model 1: Optuna Top-200 + TE]      CV Acc: {cv_m1.cv_mean_acc:.4f} +/- {cv_m1.cv_std_acc:.4f} | AUC: {cv_m1.oof_roc_auc:.4f} | LogLoss: {cv_m1.oof_log_loss:.4f} | Train: {np.mean(cv_m1.fold_train_accs):.4f}")

    cv_m2, test_probs_m2 = run_xgb_cv(
        X_df=X_train_full[pruned_features],
        y=y,
        splits=cv_splits,
        xgb_params=cfg2_params,
        experiment_id="model2_conservative_depth4_pruned_te",
        early_stopping_rounds=50,
        fold_feature_fn=fold_te_fn,
        X_test_df=X_test_full[pruned_features],
        test_fold_feature_fn=test_te_fn,
        seed=42,
    )
    print(f"[Model 2: Conservative D4 Full + TE] CV Acc: {cv_m2.cv_mean_acc:.4f} +/- {cv_m2.cv_std_acc:.4f} | AUC: {cv_m2.oof_roc_auc:.4f} | LogLoss: {cv_m2.oof_log_loss:.4f} | Train: {np.mean(cv_m2.fold_train_accs):.4f}")

    cv_m3, test_probs_m3 = run_xgb_cv(
        X_df=X_train_full[top220_features],
        y=y,
        splits=cv_splits,
        xgb_params=cfg3_obs_params,
        experiment_id="model3_obs_only_top220_no_te",
        early_stopping_rounds=50,
        fold_feature_fn=None,
        X_test_df=X_test_full[top220_features],
        test_fold_feature_fn=None,
        seed=42,
    )
    print(f"[Model 3: Pure Obs-Only Top-220]    CV Acc: {cv_m3.cv_mean_acc:.4f} +/- {cv_m3.cv_std_acc:.4f} | AUC: {cv_m3.oof_roc_auc:.4f} | LogLoss: {cv_m3.oof_log_loss:.4f} | Train: {np.mean(cv_m3.fold_train_accs):.4f}")

    cv_m4, test_probs_m4 = run_xgb_cv(
        X_df=X_train_full[top140_features],
        y=y,
        splits=cv_splits,
        xgb_params=cfg4_compact_params,
        experiment_id="model4_compact_top140_te",
        early_stopping_rounds=50,
        fold_feature_fn=fold_te_fn,
        X_test_df=X_test_full[top140_features],
        test_fold_feature_fn=test_te_fn,
        seed=42,
    )
    print(f"[Model 4: Compact Top-140 + TE]     CV Acc: {cv_m4.cv_mean_acc:.4f} +/- {cv_m4.cv_std_acc:.4f} | AUC: {cv_m4.oof_roc_auc:.4f} | LogLoss: {cv_m4.oof_log_loss:.4f} | Train: {np.mean(cv_m4.fold_train_accs):.4f}")

    # Evaluate Ensembles on Out-Of-Fold Predictions
    ens_weights = [0.15, 0.35, 0.10, 0.40]
    ens_stats = evaluate_oof_ensemble(
        y,
        [cv_m1.oof_probs, cv_m2.oof_probs, cv_m3.oof_probs, cv_m4.oof_probs],
        weights=ens_weights,
    )
    ens_oof_probs = (
        ens_weights[0] * cv_m1.oof_probs
        + ens_weights[1] * cv_m2.oof_probs
        + ens_weights[2] * cv_m3.oof_probs
        + ens_weights[3] * cv_m4.oof_probs
    )
    ens_fold_accs = [
        float(accuracy_score(y[va_idx], (ens_oof_probs[va_idx] >= 0.5).astype(int)))
        for _, va_idx in cv_splits
    ]
    print(
        f"[Ensemble (M1+M2+M3+M4 Blend)]      CV Acc: {np.mean(ens_fold_accs):.4f} +/- {np.std(ens_fold_accs):.4f} "
        f"| AUC: {ens_stats['oof_roc_auc']:.4f} | LogLoss: {ens_stats['oof_log_loss']:.4f}"
    )

    # Check threshold optimization on OOF predictions
    best_t, nested_t_acc, def_t_acc = optimize_threshold_oof(y, ens_oof_probs, cv_splits)
    print(f"Threshold check on OOF: default(0.50)={def_t_acc:.4f}, nested_tuned(best_t={best_t:.3f})={nested_t_acc:.4f}")

    # Save configurations to configs/experiments.yaml for 100% reproducibility in Kaggle notebook
    cfg_dir = ROOT / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    saved_configs = {
        "random_seed": 42,
        "n_splits": 5,
        "default_threshold": 0.5,
        "feature_subsets": {
            "pruned_features": pruned_features,
            "top200_features": top200_features,
            "top220_features": top220_features,
            "top140_features": top140_features,
        },
        "models": {
            "model1_optuna_top200_te": {
                "description": "Optuna-tuned XGBoost on Top-200 collinearity-pruned features + Within-Fold Target Encoding",
                "feature_subset": "top200_features",
                "use_within_fold_te": True,
                "early_stopping_rounds": 50,
                "xgb_params": cfg1_params,
                "cv_results": cv_m1.to_summary_dict(),
            },
            "model2_conservative_depth4_pruned_te": {
                "description": "Conservative shallow (max_depth=4) heavily regularized XGBoost on all 512 pruned features + Within-Fold TE",
                "feature_subset": "pruned_features",
                "use_within_fold_te": True,
                "early_stopping_rounds": 50,
                "xgb_params": cfg2_params,
                "cv_results": cv_m2.to_summary_dict(),
            },
            "model3_obs_only_top220_no_te": {
                "description": "Pure Observation-Only (Zero-TE) XGBoost on Top-220 features",
                "feature_subset": "top220_features",
                "use_within_fold_te": False,
                "early_stopping_rounds": 50,
                "xgb_params": cfg3_obs_params,
                "cv_results": cv_m3.to_summary_dict(),
            },
            "model4_compact_top140_te": {
                "description": "Compact Top-140 features + Within-Fold TE with depth-5 regularized trees",
                "feature_subset": "top140_features",
                "use_within_fold_te": True,
                "early_stopping_rounds": 50,
                "xgb_params": cfg4_compact_params,
                "cv_results": cv_m4.to_summary_dict(),
            },
        },
        "ensemble": {
            "description": "Weighted probability blend of diverse validated XGBoost configurations (M1 + M2 + M3 + M4)",
            "members": [
                "model1_optuna_top200_te",
                "model2_conservative_depth4_pruned_te",
                "model3_obs_only_top220_no_te",
                "model4_compact_top140_te",
            ],
            "weights": ens_weights,
            "cv_mean_acc": round(float(np.mean(ens_fold_accs)), 6),
            "cv_std_acc": round(float(np.std(ens_fold_accs)), 6),
            "fold_val_accs": [round(x, 6) for x in ens_fold_accs],
            "oof_roc_auc": ens_stats["oof_roc_auc"],
            "oof_log_loss": ens_stats["oof_log_loss"],
        },
    }
    with open(cfg_dir / "experiments.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(saved_configs, f, sort_keys=False)

    # Generate and validate the single final competition submission.csv directly from the ensemble
    assert test_probs_m1 is not None and test_probs_m2 is not None and test_probs_m3 is not None and test_probs_m4 is not None
    ens_test_probs = (
        ens_weights[0] * test_probs_m1
        + ens_weights[1] * test_probs_m2
        + ens_weights[2] * test_probs_m3
        + ens_weights[3] * test_probs_m4
    )

    sub_df = generate_submission(
        test_ids,
        (ens_test_probs >= 0.5).astype(int),
        sample_sub,
        output_path=out_dir / "submission.csv",
    )

    print("\n=== Phase 7: Validated Final Submission Summary ===")
    v_info = validate_submission_df(sub_df, sample_sub)
    print(f"  submission.csv -> {v_info}")
    print(f"Total pipeline runtime: {time.perf_counter() - t_start:.2f}s")


if __name__ == "__main__":
    main()
