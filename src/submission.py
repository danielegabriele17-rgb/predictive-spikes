"""Submission formatting and strict competition schema validation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = ["battle_id", "player_won"]


def validate_submission_df(
    sub_df: pd.DataFrame,
    sample_sub_df: pd.DataFrame,
) -> Dict[str, int | float | bool]:
    """Strictly validate a submission DataFrame against `sample_submission.csv`."""
    if list(sub_df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Invalid submission columns: expected {EXPECTED_COLUMNS}, got {list(sub_df.columns)}"
        )

    if len(sub_df) != len(sample_sub_df):
        raise ValueError(
            f"Invalid row count: expected {len(sample_sub_df)}, got {len(sub_df)}"
        )

    if sub_df["battle_id"].isna().any() or sub_df["player_won"].isna().any():
        raise ValueError("Submission contains missing (NaN) values.")

    if sub_df["battle_id"].duplicated().any():
        raise ValueError("Submission contains duplicate battle_id values.")

    if not np.array_equal(sub_df["battle_id"].to_numpy(), sample_sub_df["battle_id"].to_numpy()):
        raise ValueError("Submission battle_id ordering does not exactly match sample_submission.csv.")

    if not pd.api.types.is_integer_dtype(sub_df["player_won"]):
        raise ValueError(f"player_won column must be integer dtype, got {sub_df['player_won'].dtype}")

    unique_vals = set(sub_df["player_won"].unique().tolist())
    if not unique_vals.issubset({0, 1}):
        raise ValueError(f"player_won must contain only binary values {{0, 1}}, got {unique_vals}")

    pos_rate = float(sub_df["player_won"].mean())
    return {
        "valid": True,
        "n_rows": int(len(sub_df)),
        "positive_count": int(sub_df["player_won"].sum()),
        "negative_count": int((sub_df["player_won"] == 0).sum()),
        "positive_rate": round(pos_rate, 4),
    }


def generate_submission(
    test_battle_ids: Iterable[int],
    test_predictions: Iterable[int] | np.ndarray,
    sample_sub_df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Create, validate, and optionally save a competition submission file."""
    preds_series = pd.Series(
        np.asarray(test_predictions, dtype=int),
        index=pd.Index(test_battle_ids, name="battle_id"),
    )

    if preds_series.index.duplicated().any():
        raise ValueError("Duplicate battle_id values found in test predictions.")

    sub_df = sample_sub_df[EXPECTED_COLUMNS].copy()
    sub_df["player_won"] = sub_df["battle_id"].map(preds_series)

    if sub_df["player_won"].isna().any():
        missing_cnt = int(sub_df["player_won"].isna().sum())
        raise ValueError(f"{missing_cnt} sample_submission battle_id values missing from test predictions.")

    sub_df["player_won"] = sub_df["player_won"].astype(int)
    validate_submission_df(sub_df, sample_sub_df)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sub_df.to_csv(out, index=False)

    return sub_df
