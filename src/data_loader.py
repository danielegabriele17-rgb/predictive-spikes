"""Environment-aware data loader for local and Kaggle execution.

This module handles dataset path discovery across local development environments
and Kaggle's `/kaggle/input` read-only filesystem, parses JSONL battle logs,
and provides Generation 2 type-effectiveness lookups.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# Official competition files required for training, inference, and validation
REQUIRED_FILES = (
    "train.jsonl",
    "test.jsonl",
    "sample_submission.csv",
    "type_chart_gen2.csv",
)


def discover_data_dir(custom_dir: Optional[str | Path] = None) -> Path:
    """Discover the directory containing competition dataset files.

    Searches in order:
    1. Explicit `custom_dir` argument or `POKEMON_DATA_DIR` environment variable.
    2. Local repository `fds-pokemon-battles-prediction-2026` directory.
    3. `/kaggle/input` recursive search (works regardless of Kaggle slug mounting).
    """
    candidates: List[Path] = []

    if custom_dir is not None:
        candidates.append(Path(custom_dir))

    env_dir = os.environ.get("POKEMON_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    repo_root = Path(__file__).resolve().parent.parent
    candidates.append(repo_root / "fds-pokemon-battles-prediction-2026")
    candidates.append(Path.cwd() / "fds-pokemon-battles-prediction-2026")

    # Return the first candidate directory that contains the training set
    for cand in candidates:
        if cand.is_dir() and (cand / "train.jsonl").is_file():
            return cand.resolve()

    # Fallback: recursively scan Kaggle's input directory
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for match in kaggle_input.rglob("train.jsonl"):
            if match.is_file():
                return match.parent.resolve()

    raise FileNotFoundError(
        "Could not locate competition data directory containing 'train.jsonl'. "
        f"Checked candidates: {[str(c) for c in candidates]} and /kaggle/input."
    )


def get_data_paths(custom_dir: Optional[str | Path] = None) -> Dict[str, Path]:
    """Return validated paths to all competition dataset files."""
    data_dir = discover_data_dir(custom_dir)
    paths: Dict[str, Path] = {"data_dir": data_dir}
    for fname in REQUIRED_FILES:
        fpath = data_dir / fname
        if not fpath.is_file():
            raise FileNotFoundError(f"Missing required competition file: {fpath}")
        paths[fname.split(".")[0]] = fpath
    return paths


def load_jsonl(path: str | Path, max_records: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load a JSONL file line by line into a list of dictionaries."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_records is not None and idx >= max_records:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_type_chart(path: Optional[str | Path] = None) -> Dict[Tuple[str, str], float]:
    """Load the Generation 2 type chart into a `(attacking_type, defending_type) -> multiplier` dict.

    Uses vectorized pandas string normalization. Pairs not listed in `type_chart_gen2.csv`
    default to neutral effectiveness (1.0), and `'notype'` is always neutral (1.0).
    """
    if path is None:
        path = get_data_paths()["type_chart_gen2"]
    df = pd.read_csv(path)
    atk_types = df["attacking_type"].astype(str).str.strip().str.lower()
    def_types = df["defending_type"].astype(str).str.strip().str.lower()
    multipliers = df["multiplier"].astype(float)
    return dict(zip(zip(atk_types, def_types), multipliers))


def get_type_effectiveness(
    atk_type: Optional[str],
    def_types: List[str] | Tuple[str, ...],
    type_chart: Dict[Tuple[str, str], float],
) -> float:
    """Compute Generation 2 type effectiveness multiplier of `atk_type` against `def_types`.

    For dual-type defending Pokémon (e.g., Rock/Ground), individual type multipliers
    are multiplied together via `math.prod` (yielding values in {0.0, 0.25, 0.5, 1.0, 2.0, 4.0}).
    """
    if not atk_type or atk_type in ("notype", "threequestionmarks", "none"):
        return 1.0
    atk_t = atk_type.lower()
    return math.prod(
        type_chart.get((atk_t, dt.lower()), 1.0)
        for dt in def_types
        if dt and dt != "notype"
    )


def load_competition_data(
    custom_dir: Optional[str | Path] = None,
    max_records: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame, Dict[Tuple[str, str], float]]:
    """Load train battles, test battles, sample_submission DataFrame, and Gen 2 type chart."""
    paths = get_data_paths(custom_dir)
    train_battles = load_jsonl(paths["train"], max_records=max_records)
    test_battles = load_jsonl(paths["test"], max_records=max_records)
    sample_sub = pd.read_csv(paths["sample_submission"])
    type_chart = load_type_chart(paths["type_chart_gen2"])
    return train_battles, test_battles, sample_sub, type_chart
