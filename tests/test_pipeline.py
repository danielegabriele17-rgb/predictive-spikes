"""Automated verification test suite for the Pokémon Battle Prediction submission pipeline."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.battle_features import (
    build_species_catalogue,
    reconstruct_battle_history,
)
from src.data_loader import (
    get_data_paths,
    get_type_effectiveness,
    load_jsonl,
    load_type_chart,
)
from src.feature_engineering import (
    ALL_FEATURE_FAMILIES,
    WithinFoldTargetEncoder,
    build_feature_dataframe,
)
from src.model import get_feature_importance_df, optimize_ensemble_weights, run_xgb_cv
from src.submission import generate_submission, validate_submission_df
from src.validation import get_cv_splits


@pytest.fixture(scope="module")
def sample_data():
    paths = get_data_paths()
    train_small = load_jsonl(paths["train"], max_records=60)
    test_small = load_jsonl(paths["test"], max_records=25)
    sample_sub = pd.read_csv(paths["sample_submission"]).iloc[:25].copy()
    type_chart = load_type_chart(paths["type_chart_gen2"])
    catalogue = build_species_catalogue(train_small)
    return train_small, test_small, sample_sub, type_chart, catalogue


def test_data_loader_and_schema(sample_data):
    train_small, test_small, sample_sub, type_chart, _ = sample_data
    assert len(train_small) == 60
    assert len(test_small) == 25
    assert "player_won" in train_small[0]
    assert "player_won" not in test_small[0]
    assert len(train_small[0]["p1_team_details"]) == 6
    assert len(train_small[0]["battle_timeline"]) == 30

    # Verify Gen 2 type chart multipliers
    assert get_type_effectiveness("ground", ["electric"], type_chart) == 2.0
    assert get_type_effectiveness("electric", ["ground"], type_chart) == 0.0
    assert get_type_effectiveness("fire", ["grass", "steel"], type_chart) == 4.0
    assert get_type_effectiveness("normal", ["notype"], type_chart) == 1.0


def test_duplicate_cached_move_not_double_counted(sample_data):
    """Verify that when a Pokémon switches in after having moved earlier,
    its cached `move_details` is NOT double-counted as a newly executed attack.
    """
    train_small, _, _, type_chart, catalogue = sample_data
    battle = copy.deepcopy(train_small[0])

    tl = battle["battle_timeline"][:3]
    tl[0]["p1_pokemon_state"] = {"name": "tyranitar", "hp_pct": 1.0, "status": "nostatus", "effects": ["noeffect"], "boosts": {}}
    tl[0]["p1_move_details"] = {"name": "earthquake", "type": "ground", "category": "physical", "base_power": 100, "accuracy": 1.0, "priority": 0}

    tl[1]["p1_pokemon_state"] = {"name": "golem", "hp_pct": 1.0, "status": "nostatus", "effects": ["noeffect"], "boosts": {}}
    tl[1]["p1_move_details"] = None

    tl[2]["p1_pokemon_state"] = {"name": "tyranitar", "hp_pct": 1.0, "status": "nostatus", "effects": ["noeffect"], "boosts": {}}
    tl[2]["p1_move_details"] = {"name": "earthquake", "type": "ground", "category": "physical", "base_power": 100, "accuracy": 1.0, "priority": 0}

    battle["battle_timeline"] = tl
    recon = reconstruct_battle_history(battle, catalogue, type_chart)
    assert recon["p1_attacked_turn"][0] == 1.0
    assert recon["p1_attacked_turn"][1] == 0.0
    assert recon["p1_attacked_turn"][2] == 0.0
    assert recon["p1_switched_turn"][2] == 1.0


def test_missing_values_and_observation_boundaries(sample_data):
    train_small, _, _, type_chart, catalogue = sample_data
    b_missing = copy.deepcopy(train_small[0])
    b_missing["p2_lead_details"] = {"name": "unseenmon"}
    b_missing["battle_timeline"] = b_missing["battle_timeline"][:12]
    for e in b_missing["battle_timeline"]:
        e["p1_move_details"] = None
        e["p2_move_details"] = None

    X_df, _, _ = build_feature_dataframe([b_missing], families=ALL_FEATURE_FAMILIES, catalogue=catalogue, type_chart=type_chart)
    assert X_df.shape[0] == 1
    assert not X_df.isna().any().any()
    assert not np.isinf(X_df.to_numpy()).any()


def test_train_test_feature_consistency_and_no_target_leakage(sample_data):
    train_small, test_small, _, type_chart, catalogue = sample_data
    X_tr, _, y_tr = build_feature_dataframe(train_small, families=ALL_FEATURE_FAMILIES, catalogue=catalogue, type_chart=type_chart)
    X_te, _, y_te = build_feature_dataframe(test_small, families=ALL_FEATURE_FAMILIES, catalogue=catalogue, type_chart=type_chart)

    assert y_tr is not None and len(y_tr) == len(train_small)
    assert y_te is None
    assert list(X_tr.columns) == list(X_te.columns)
    assert "battle_id" not in X_tr.columns
    assert "player_won" not in X_tr.columns

    # Mutate player_won in train_small and verify extracted features are 100% invariant
    train_flipped = copy.deepcopy(train_small)
    for b in train_flipped:
        b["player_won"] = 1 - int(b["player_won"])
    X_tr_flipped, _, _ = build_feature_dataframe(train_flipped, families=ALL_FEATURE_FAMILIES, catalogue=catalogue, type_chart=type_chart)
    pd.testing.assert_frame_equal(X_tr, X_tr_flipped)


def test_within_fold_target_encoder_no_leakage(sample_data):
    train_small, _, _, _, _ = sample_data
    y = np.array([int(b["player_won"]) for b in train_small], dtype=int)
    enc = WithinFoldTargetEncoder(smoothing=20.0, team_smoothing=15.0)
    tr_oof = enc.fit_transform_inner_oof(train_small[:40], y[:40], n_inner_splits=4, seed=42)
    va_df = enc.transform(train_small[40:])
    assert len(tr_oof) == 40
    assert len(va_df) == 20
    assert list(tr_oof.columns) == list(va_df.columns)
    assert not tr_oof.isna().any().any()


def test_model_cv_and_submission_formatting(sample_data, tmp_path):
    train_small, test_small, sample_sub, type_chart, catalogue = sample_data
    X_tr, _, y_tr = build_feature_dataframe(train_small, families=["team", "state"], catalogue=catalogue, type_chart=type_chart)
    X_te, te_ids, _ = build_feature_dataframe(test_small, families=["team", "state"], catalogue=catalogue, type_chart=type_chart)

    splits = get_cv_splits(train_small, y_tr, n_splits=3, seed=42)
    fast_params = {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1, "n_jobs": 2}
    cv_res, test_probs = run_xgb_cv(
        X_df=X_tr,
        y=y_tr,
        splits=splits,
        xgb_params=fast_params,
        experiment_id="unit_test_xgb",
        early_stopping_rounds=10,
        X_test_df=X_te,
    )
    assert 0.0 <= cv_res.cv_mean_acc <= 1.0
    assert test_probs is not None and len(test_probs) == len(test_small)

    fi_df = get_feature_importance_df(cv_res.models, cv_res.feature_names)
    assert {"feature", "importance", "gain", "total_gain", "weight", "cover"}.issubset(set(fi_df.columns))

    test_preds = (test_probs >= 0.5).astype(int)
    out_csv = tmp_path / "test_sub.csv"
    sub_df = generate_submission(te_ids, test_preds, sample_sub, output_path=out_csv)
    assert out_csv.is_file()
    info = validate_submission_df(sub_df, sample_sub)
    assert info["valid"] is True
    assert info["n_rows"] == len(sample_sub)


def test_optimize_ensemble_weights():
    np.random.seed(42)
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 1])
    # 3 mock model prediction columns
    p1 = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3, 0.8, 0.2, 0.9, 0.6])
    p2 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    p3 = np.array([0.8, 0.7, 0.2, 0.3, 0.6, 0.4, 0.7, 0.1, 0.8, 0.7])
    P = np.column_stack([p1, p2, p3])

    opt_w = optimize_ensemble_weights(y_true, P, init_weights=[0.33, 0.33, 0.33], metric="logloss")
    assert len(opt_w) == 3
    assert np.isclose(np.sum(opt_w), 1.0)
    assert all(w >= 0.0 for w in opt_w)
    # The informative models (p1, p3) should receive much higher weight than the random guesser p2
    assert opt_w[0] > opt_w[1] and opt_w[2] > opt_w[1]

    # Test accuracy metric
    opt_w_acc = optimize_ensemble_weights(y_true, P, metric="accuracy")
    assert len(opt_w_acc) == 3
    assert np.isclose(np.sum(opt_w_acc), 1.0)

