"""Run progressive baseline and feature ablation experiments with 5-fold Stratified CV."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

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
from src.validation import get_cv_splits


def run_ablation_suite() -> None:
    print("Loading competition dataset...")
    t0 = time.perf_counter()
    train_battles, test_battles, sample_sub, type_chart = load_competition_data()
    print(f"Loaded {len(train_battles)} train battles and {len(test_battles)} test battles in {time.perf_counter() - t0:.2f}s.")

    y = np.array([int(b["player_won"]) for b in train_battles], dtype=int)
    splits = get_cv_splits(train_battles, y, n_splits=5, seed=42, group_duplicates=False)
    splits_grouped = get_cv_splits(train_battles, y, n_splits=5, seed=42, group_duplicates=True)

    # Pre-extract each individual family once so combinations are instantaneous
    catalogue = build_species_catalogue(train_battles)
    print("\nExtracting modular feature families on 10,000 training battles...")
    t_ext = time.perf_counter()
    X_org, _, _ = build_feature_dataframe(train_battles, families=["organizer"], catalogue=catalogue, type_chart=type_chart)
    X_simple, _, _ = build_feature_dataframe(train_battles, families=["team", "simple_state"], catalogue=catalogue, type_chart=type_chart)
    X_full, _, _ = build_feature_dataframe(train_battles, families=ALL_FEATURE_FAMILIES, catalogue=catalogue, type_chart=type_chart)
    ext_time = time.perf_counter() - t_ext
    print(f"Feature extraction complete in {ext_time:.2f}s. Total engineered features in X_full: {X_full.shape[1]}")

    # Map columns by family prefix for clean slicing
    team_cols = list(build_feature_dataframe(train_battles[:5], families=["team"], catalogue=catalogue, type_chart=type_chart)[0].columns)
    p2_obs_cols = [c for c in build_feature_dataframe(train_battles[:5], families=["p2_observed"], catalogue=catalogue, type_chart=type_chart)[0].columns]
    state_cols = [c for c in build_feature_dataframe(train_battles[:5], families=["state"], catalogue=catalogue, type_chart=type_chart)[0].columns]
    temp_cols = [c for c in build_feature_dataframe(train_battles[:5], families=["temporal"], catalogue=catalogue, type_chart=type_chart)[0].columns]
    match_cols = [c for c in build_feature_dataframe(train_battles[:5], families=["matchup"], catalogue=catalogue, type_chart=type_chart)[0].columns]
    inter_cols = [c for c in X_full.columns if c.startswith("inter_")]

    print(
        f"Family feature counts: team={len(team_cols)}, p2_observed={len(p2_obs_cols)}, "
        f"state={len(state_cols)}, temporal={len(temp_cols)}, matchup={len(match_cols)}, interactions={len(inter_cols)}"
    )

    results_log = []

    # Exp 00: Official Organizer Baseline (LogisticRegression on 6 features)
    t_lr = time.perf_counter()
    lr_oof = np.zeros(len(y), dtype=float)
    lr_val_accs, lr_tr_accs = [], []
    for tr_idx, va_idx in splits:
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_org.iloc[tr_idx], y[tr_idx])
        lr_tr_accs.append(float(accuracy_score(y[tr_idx], lr.predict(X_org.iloc[tr_idx]))))
        probs = lr.predict_proba(X_org.iloc[va_idx])[:, 1]
        lr_oof[va_idx] = probs
        lr_val_accs.append(float(accuracy_score(y[va_idx], (probs >= 0.5).astype(int))))
    lr_summary = {
        "experiment_id": "exp00_organizer_logreg",
        "families": ["organizer_6"],
        "model": "LogisticRegression(max_iter=1000)",
        "cv_mean_acc": round(float(np.mean(lr_val_accs)), 6),
        "cv_std_acc": round(float(np.std(lr_val_accs)), 6),
        "oof_accuracy": round(float(accuracy_score(y, (lr_oof >= 0.5).astype(int))), 6),
        "oof_roc_auc": round(float(roc_auc_score(y, lr_oof)), 6),
        "oof_log_loss": round(float(log_loss(y, lr_oof)), 6),
        "train_mean_acc": round(float(np.mean(lr_tr_accs)), 6),
        "train_val_gap": round(float(np.mean(lr_tr_accs) - np.mean(lr_val_accs)), 6),
        "fold_val_accs": [round(x, 6) for x in lr_val_accs],
        "n_features": X_org.shape[1],
        "runtime_sec": round(time.perf_counter() - t_lr, 3),
    }
    results_log.append(lr_summary)
    print(f"[exp00_organizer_logreg] CV Acc: {lr_summary['cv_mean_acc']:.4f} +/- {lr_summary['cv_std_acc']:.4f} | Features: {lr_summary['n_features']}")

    # Progressive XGBoost experiments
    experiments = [
        ("exp01_organizer_xgb", ["organizer_6"], X_org),
        ("exp02_xgb_initial_baseline", ["team", "simple_state"], X_simple),
        ("exp03_ablation_team_only", ["team"], X_full[team_cols]),
        ("exp04_ablation_team_state", ["team", "state"], X_full[team_cols + state_cols]),
        ("exp05_ablation_team_p2_state", ["team", "p2_observed", "state"], X_full[team_cols + p2_obs_cols + state_cols]),
        ("exp06_ablation_plus_temporal", ["team", "p2_observed", "state", "temporal"], X_full[team_cols + p2_obs_cols + state_cols + temp_cols]),
        ("exp07_ablation_plus_matchup", ["team", "p2_observed", "state", "temporal", "matchup"], X_full[team_cols + p2_obs_cols + state_cols + temp_cols + match_cols]),
        ("exp08_ablation_all_families", list(ALL_FEATURE_FAMILIES), X_full),
    ]

    for exp_id, fams, df_sub in experiments:
        cv_res, _ = run_xgb_cv(
            X_df=df_sub,
            y=y,
            splits=splits,
            xgb_params=DEFAULT_XGB_PARAMS,
            experiment_id=exp_id,
            early_stopping_rounds=50,
            seed=42,
        )
        summary = cv_res.to_summary_dict()
        summary["families"] = fams
        summary["model"] = "XGBClassifier(default_baseline)"
        results_log.append(summary)
        print(
            f"[{exp_id:28s}] CV Acc: {cv_res.cv_mean_acc:.4f} +/- {cv_res.cv_std_acc:.4f} "
            f"| AUC: {cv_res.oof_roc_auc:.4f} | Train: {summary['train_mean_acc']:.4f} "
            f"| Features: {cv_res.n_features:3d} | Time: {cv_res.runtime_sec:.2f}s"
        )

    # Exp 09: All families + Within-Fold Target Encoding
    def fold_te_fn(tr_idx: np.ndarray, va_idx: np.ndarray):
        enc = WithinFoldTargetEncoder(smoothing=30.0, team_smoothing=20.0)
        tr_b = [train_battles[i] for i in tr_idx]
        va_b = [train_battles[i] for i in va_idx]
        tr_te = enc.fit_transform_inner_oof(tr_b, y[tr_idx], n_inner_splits=5, seed=42)
        va_te = enc.transform(va_b)
        return tr_te, va_te

    cv_res_te, _ = run_xgb_cv(
        X_df=X_full,
        y=y,
        splits=splits,
        xgb_params=DEFAULT_XGB_PARAMS,
        experiment_id="exp09_all_plus_within_fold_te",
        early_stopping_rounds=50,
        fold_feature_fn=fold_te_fn,
        seed=42,
    )
    summary_te = cv_res_te.to_summary_dict()
    summary_te["families"] = list(ALL_FEATURE_FAMILIES) + ["within_fold_te"]
    summary_te["model"] = "XGBClassifier(default_baseline)"
    results_log.append(summary_te)
    print(
        f"[{'exp09_all_plus_within_fold_te':28s}] CV Acc: {cv_res_te.cv_mean_acc:.4f} +/- {cv_res_te.cv_std_acc:.4f} "
        f"| AUC: {cv_res_te.oof_roc_auc:.4f} | Train: {summary_te['train_mean_acc']:.4f} "
        f"| Features: {cv_res_te.n_features:3d} | Time: {cv_res_te.runtime_sec:.2f}s"
    )

    # Also check Exp 08 on duplicate-grouped splits
    cv_res_grp, _ = run_xgb_cv(
        X_df=X_full,
        y=y,
        splits=splits_grouped,
        xgb_params=DEFAULT_XGB_PARAMS,
        experiment_id="exp08b_all_families_grouped_cv",
        early_stopping_rounds=50,
        seed=42,
    )
    summary_grp = cv_res_grp.to_summary_dict()
    summary_grp["families"] = list(ALL_FEATURE_FAMILIES)
    summary_grp["model"] = "XGBClassifier(StratifiedGroupKFold)"
    results_log.append(summary_grp)
    print(
        f"[{'exp08b_all_families_grouped_cv':28s}] CV Acc: {cv_res_grp.cv_mean_acc:.4f} +/- {cv_res_grp.cv_std_acc:.4f} "
        f"| AUC: {cv_res_grp.oof_roc_auc:.4f} | Features: {cv_res_grp.n_features:3d}"
    )

    # Save experiment log and top feature importances
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=2)

    imp_df = get_feature_importance_df(cv_res_grp.models, cv_res_grp.feature_names)
    imp_df.to_csv(out_dir / "feature_importance_all.csv", index=False)
    print("\nTop 25 most important features (exp08):")
    print(imp_df.head(25).to_string(index=False))


if __name__ == "__main__":
    run_ablation_suite()
