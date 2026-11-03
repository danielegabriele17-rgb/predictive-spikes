"""Reproduce the three selected competition submissions from version-controlled configurations."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
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
from src.model import run_xgb_cv
from src.submission import generate_submission, validate_submission_df
from src.validation import get_cv_splits


def reproduce_all_submissions(
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> Dict[str, pd.DataFrame]:
    """Load `configs/experiments.yaml`, retrain the selected XGBoost models via 5-fold CV,
    and generate the three validated competition submission CSV files.
    """
    t0 = time.perf_counter()
    if config_path is None:
        config_path = ROOT / "configs" / "experiments.yaml"
    if output_dir is None:
        output_dir = ROOT / "outputs"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("random_seed", 42))
    n_splits = int(cfg.get("n_splits", 5))
    threshold = float(cfg.get("default_threshold", 0.5))

    print("1. Loading competition dataset...")
    train_battles, test_battles, sample_sub, type_chart = load_competition_data(custom_dir=data_dir)
    y = np.array([int(b["player_won"]) for b in train_battles], dtype=int)
    catalogue = build_species_catalogue(train_battles)

    print("2. Extracting feature families...")
    X_train_full, _, _ = build_feature_dataframe(
        train_battles, families=ALL_FEATURE_FAMILIES, catalogue=catalogue, type_chart=type_chart
    )
    X_test_full, test_ids, _ = build_feature_dataframe(
        test_battles, families=ALL_FEATURE_FAMILIES, catalogue=catalogue, type_chart=type_chart
    )

    cv_splits = get_cv_splits(train_battles, y, n_splits=n_splits, seed=seed, group_duplicates=False)

    print("3. Computing strictly within-fold leak-free target encodings...")
    te_folds_train: List[pd.DataFrame] = []
    te_folds_val: List[pd.DataFrame] = []
    te_folds_test: List[pd.DataFrame] = []
    for tr_idx, va_idx in cv_splits:
        enc = WithinFoldTargetEncoder(smoothing=30.0, team_smoothing=20.0)
        tr_b = [train_battles[i] for i in tr_idx]
        va_b = [train_battles[i] for i in va_idx]
        te_folds_train.append(enc.fit_transform_inner_oof(tr_b, y[tr_idx], n_inner_splits=5, seed=seed))
        te_folds_val.append(enc.transform(va_b))
        te_folds_test.append(enc.transform(test_battles))

    fold_map = {(int(tr[0]), int(va[0])): idx for idx, (tr, va) in enumerate(cv_splits)}

    def fold_te_fn(tr_idx: np.ndarray, va_idx: np.ndarray):
        idx = fold_map[(int(tr_idx[0]), int(va_idx[0]))]
        return te_folds_train[idx], te_folds_val[idx]

    def test_te_fn(tr_idx: np.ndarray, va_idx: np.ndarray):
        idx = fold_map[(int(tr_idx[0]), int(va_idx[0]))]
        return te_folds_test[idx]

    subsets = cfg["feature_subsets"]
    models_cfg = cfg["models"]

    oof_map: Dict[str, np.ndarray] = {}
    test_map: Dict[str, np.ndarray] = {}

    print("4. Training and validating selected XGBoost configurations...")
    for m_name, m_spec in models_cfg.items():
        cols = subsets[m_spec["feature_subset"]]
        use_te = bool(m_spec.get("use_within_fold_te", True))
        es_rounds = m_spec.get("early_stopping_rounds", 50)
        params = m_spec["xgb_params"]

        cv_res, test_probs = run_xgb_cv(
            X_df=X_train_full[cols],
            y=y,
            splits=cv_splits,
            xgb_params=params,
            experiment_id=m_name,
            early_stopping_rounds=es_rounds,
            fold_feature_fn=fold_te_fn if use_te else None,
            X_test_df=X_test_full[cols],
            test_fold_feature_fn=test_te_fn if use_te else None,
            threshold=threshold,
            seed=seed,
        )
        assert test_probs is not None
        oof_map[m_name] = cv_res.oof_probs
        test_map[m_name] = test_probs
        print(
            f"   [{m_name:36s}] CV Acc: {cv_res.cv_mean_acc:.4f} +/- {cv_res.cv_std_acc:.4f} "
            f"| AUC: {cv_res.oof_roc_auc:.4f} | LogLoss: {cv_res.oof_log_loss:.4f}"
        )

    ens_cfg = cfg["ensemble"]
    members = ens_cfg["members"]
    weights = np.asarray(ens_cfg["weights"], dtype=float)
    weights = weights / np.sum(weights)

    ens_oof = sum(w * oof_map[m] for w, m in zip(weights, members))
    ens_test = sum(w * test_map[m] for w, m in zip(weights, members))
    ens_fold_accs = [
        float(accuracy_score(y[va_idx], (ens_oof[va_idx] >= threshold).astype(int)))
        for _, va_idx in cv_splits
    ]
    print(
        f"   [{'ensemble_diverse_xgb_blend':36s}] CV Acc: {np.mean(ens_fold_accs):.4f} +/- {np.std(ens_fold_accs):.4f} "
        f"| AUC: {roc_auc_score(y, ens_oof):.4f} | LogLoss: {log_loss(y, ens_oof):.4f}"
    )

    print("5. Generating and validating submission CSV files...")
    sub1 = generate_submission(
        test_ids,
        (test_map["model4_compact_top140_te"] >= threshold).astype(int),
        sample_sub,
        output_path=out_path / "submission_1_compact_top140_xgb.csv",
    )
    sub2 = generate_submission(
        test_ids,
        (test_map["model2_conservative_depth4_pruned_te"] >= threshold).astype(int),
        sample_sub,
        output_path=out_path / "submission_2_conservative_depth4_xgb.csv",
    )
    sub3 = generate_submission(
        test_ids,
        (ens_test >= threshold).astype(int),
        sample_sub,
        output_path=out_path / "submission_3_diverse_xgb_ensemble.csv",
    )
    generate_submission(
        test_ids,
        (ens_test >= threshold).astype(int),
        sample_sub,
        output_path=out_path / "submission.csv",
    )

    for label, df_s in [
        ("submission_1_compact_top140_xgb.csv", sub1),
        ("submission_2_conservative_depth4_xgb.csv", sub2),
        ("submission_3_diverse_xgb_ensemble.csv", sub3),
    ]:
        info = validate_submission_df(df_s, sample_sub)
        print(f"   Validated {label:42s}: {info}")

    print(f"Done in {time.perf_counter() - t0:.2f}s.")
    return {
        "submission_1": sub1,
        "submission_2": sub2,
        "submission_3": sub3,
    }


if __name__ == "__main__":
    reproduce_all_submissions()
