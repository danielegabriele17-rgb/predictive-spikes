"""Configurable feature engineering pipeline and within-fold leak-free encoders."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.battle_features import (
    build_species_catalogue,
    extract_interaction_features,
    extract_matchup_features,
    extract_organizer_features,
    extract_p2_observed_features,
    extract_simple_state_features,
    extract_state_features,
    extract_team_features,
    extract_temporal_features,
    reconstruct_battle_history,
)


ALL_FEATURE_FAMILIES: Tuple[str, ...] = (
    "team",
    "p2_observed",
    "state",
    "temporal",
    "matchup",
    "interactions",
)


def extract_battle_features_single(
    battle: Dict[str, Any],
    families: Sequence[str],
    catalogue: Dict[str, Dict[str, Any]],
    type_chart: Dict[Tuple[str, str], float],
) -> Dict[str, float]:
    """Extract requested feature families for a single battle record."""
    fam_set = set(families)
    feats: Dict[str, float] = {}

    if "organizer" in fam_set:
        feats.update(extract_organizer_features(battle))
        return feats

    if "team" in fam_set:
        feats.update(extract_team_features(battle))

    if "simple_state" in fam_set:
        feats.update(extract_simple_state_features(battle))

    needs_recon = bool(fam_set & {"p2_observed", "state", "temporal", "matchup", "interactions"})
    if needs_recon:
        recon = reconstruct_battle_history(battle, catalogue, type_chart)
        if "p2_observed" in fam_set:
            feats.update(extract_p2_observed_features(battle, recon, catalogue))
        if "state" in fam_set or "interactions" in fam_set:
            st_feats = extract_state_features(battle, recon, catalogue)
            if "state" in fam_set:
                feats.update(st_feats)
        if "temporal" in fam_set or "interactions" in fam_set:
            tmp_feats = extract_temporal_features(battle, recon)
            if "temporal" in fam_set:
                feats.update(tmp_feats)
        if "matchup" in fam_set or "interactions" in fam_set:
            mt_feats = extract_matchup_features(battle, recon, catalogue, type_chart)
            if "matchup" in fam_set:
                feats.update(mt_feats)
        if "interactions" in fam_set:
            merged_context = {}
            merged_context.update(feats)
            if "state" not in fam_set:
                merged_context.update(st_feats)
            if "temporal" not in fam_set:
                merged_context.update(tmp_feats)
            if "matchup" not in fam_set:
                merged_context.update(mt_feats)
            feats.update(extract_interaction_features(merged_context))

    return feats


def build_feature_dataframe(
    battles: List[Dict[str, Any]],
    families: Sequence[str] = ALL_FEATURE_FAMILIES,
    catalogue: Optional[Dict[str, Dict[str, Any]]] = None,
    type_chart: Optional[Dict[Tuple[str, str], float]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]:
    """Transform a list of battle dicts into `(X_df, battle_ids, y_or_None)`.

    All features returned by `build_feature_dataframe` are purely observation-based
    (zero dependence on the target `player_won`).
    """
    if catalogue is None:
        catalogue = build_species_catalogue(battles)
    if type_chart is None:
        type_chart = {}

    # Extract battle IDs, optional binary targets, and observation-only feature rows
    battle_ids = np.array([int(b.get("battle_id", idx)) for idx, b in enumerate(battles)], dtype=int)
    has_target = len(battles) > 0 and ("player_won" in battles[0])
    y = np.array([int(b["player_won"]) for b in battles], dtype=int) if has_target else None
    rows = [extract_battle_features_single(b, families, catalogue, type_chart) for b in battles]

    X_df = pd.DataFrame(rows)
    return X_df, battle_ids, y


class WithinFoldTargetEncoder:
    """Strictly within-fold smoothed Bayesian target encoder for Pokémon species,
    surviving species, and team archetype signatures.

    Never sees validation labels during fit; uses Laplace/empirical-Bayes smoothing
    toward the training fold's global prior mean.
    """

    def __init__(self, smoothing: float = 30.0, team_smoothing: float = 20.0) -> None:
        self.smoothing = float(smoothing)
        self.team_smoothing = float(team_smoothing)
        self.global_prior: float = 0.5
        self.p1_species_rate: Dict[str, float] = {}
        self.p2_species_rate: Dict[str, float] = {}
        self.p1_alive_species_rate: Dict[str, float] = {}
        self.p2_alive_species_rate: Dict[str, float] = {}
        self.p1_team_rate: Dict[Tuple[str, ...], float] = {}
        self.p2_team_rate: Dict[Tuple[str, ...], float] = {}

    @staticmethod
    def _extract_battle_tokens(battle: Dict[str, Any]) -> Dict[str, Any]:
        p1_names = [p["name"] for p in battle.get("p1_team_details", [])]
        p1_hp = {n: 1.0 for n in p1_names}
        p2_lead = (battle.get("p2_lead_details") or {}).get("name", "")
        p2_hp = {p2_lead: 1.0} if p2_lead else {}

        for e in battle.get("battle_timeline", []):
            s1 = e["p1_pokemon_state"]
            s2 = e["p2_pokemon_state"]
            p1_hp[s1["name"]] = 0.0 if s1["status"] == "fnt" else float(s1["hp_pct"])
            p2_hp[s2["name"]] = 0.0 if s2["status"] == "fnt" else float(s2["hp_pct"])

        p1_alive = [n for n, h in p1_hp.items() if h > 0.0]
        p2_alive = [n for n, h in p2_hp.items() if h > 0.0]
        p1_sig = tuple(sorted(p1_names))
        p2_sig = tuple(sorted(p2_hp.keys()))
        tl = battle.get("battle_timeline", [])
        act_p1 = tl[-1]["p1_pokemon_state"]["name"] if tl else ""
        act_p2 = tl[-1]["p2_pokemon_state"]["name"] if tl else ""
        return {
            "p1_names": p1_names,
            "p2_names": list(p2_hp.keys()),
            "p1_alive": p1_alive,
            "p2_alive": p2_alive,
            "p1_hp": p1_hp,
            "p2_hp": p2_hp,
            "p1_sig": p1_sig,
            "p2_sig": p2_sig,
            "act_p1": act_p1,
            "act_p2": act_p2,
        }

    def fit(self, train_battles: Sequence[Dict[str, Any]], y_train: np.ndarray) -> "WithinFoldTargetEncoder":
        self.global_prior = float(np.mean(y_train)) if len(y_train) > 0 else 0.5

        p1_sp_stats: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])
        p2_sp_stats: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])
        p1_alv_stats: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])
        p2_alv_stats: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])
        p1_tm_stats: Dict[Tuple[str, ...], List[float]] = defaultdict(lambda: [0.0, 0.0])
        p2_tm_stats: Dict[Tuple[str, ...], List[float]] = defaultdict(lambda: [0.0, 0.0])

        for b, target in zip(train_battles, y_train):
            t_val = float(target)
            tok = self._extract_battle_tokens(b)
            for n in tok["p1_names"]:
                p1_sp_stats[n][0] += t_val
                p1_sp_stats[n][1] += 1.0
            for n in tok["p2_names"]:
                p2_sp_stats[n][0] += t_val
                p2_sp_stats[n][1] += 1.0
            for n in tok["p1_alive"]:
                p1_alv_stats[n][0] += t_val
                p1_alv_stats[n][1] += 1.0
            for n in tok["p2_alive"]:
                p2_alv_stats[n][0] += t_val
                p2_alv_stats[n][1] += 1.0
            p1_tm_stats[tok["p1_sig"]][0] += t_val
            p1_tm_stats[tok["p1_sig"]][1] += 1.0
            if len(tok["p2_sig"]) == 6:
                p2_tm_stats[tok["p2_sig"]][0] += t_val
                p2_tm_stats[tok["p2_sig"]][1] += 1.0

        m = self.smoothing
        mt = self.team_smoothing
        g = self.global_prior

        self.p1_species_rate = {k: (v[0] + m * g) / (v[1] + m) for k, v in p1_sp_stats.items()}
        self.p2_species_rate = {k: (v[0] + m * g) / (v[1] + m) for k, v in p2_sp_stats.items()}
        self.p1_alive_species_rate = {k: (v[0] + m * g) / (v[1] + m) for k, v in p1_alv_stats.items()}
        self.p2_alive_species_rate = {k: (v[0] + m * g) / (v[1] + m) for k, v in p2_alv_stats.items()}
        self.p1_team_rate = {k: (v[0] + mt * g) / (v[1] + mt) for k, v in p1_tm_stats.items()}
        self.p2_team_rate = {k: (v[0] + mt * g) / (v[1] + mt) for k, v in p2_tm_stats.items()}
        return self

    def transform_single(self, battle: Dict[str, Any]) -> Dict[str, float]:
        g = self.global_prior
        tok = self._extract_battle_tokens(battle)

        p1_rates = [self.p1_species_rate.get(n, g) for n in tok["p1_names"]]
        p2_rates = [self.p2_species_rate.get(n, g) for n in tok["p2_names"]]
        p1_alv_rates = [self.p1_alive_species_rate.get(n, g) for n in tok["p1_alive"]]
        p2_alv_rates = [self.p2_alive_species_rate.get(n, g) for n in tok["p2_alive"]]

        p1_sp_mean = float(np.mean(p1_rates)) if p1_rates else g
        p2_sp_mean = float(np.mean(p2_rates)) if p2_rates else g
        p1_alv_mean = float(np.mean(p1_alv_rates)) if p1_alv_rates else g
        p2_alv_mean = float(np.mean(p2_alv_rates)) if p2_alv_rates else g

        p1_hp_map = tok["p1_hp"]
        p2_hp_map = tok["p2_hp"]
        p1_hp_sum = sum(p1_hp_map.values())
        p2_hp_sum = sum(p2_hp_map.values())
        p1_hp_wt = (
            sum(h * self.p1_alive_species_rate.get(n, g) for n, h in p1_hp_map.items() if h > 0) / p1_hp_sum
            if p1_hp_sum > 0 else g
        )
        p2_hp_wt = (
            sum(h * self.p2_alive_species_rate.get(n, g) for n, h in p2_hp_map.items() if h > 0) / p2_hp_sum
            if p2_hp_sum > 0 else g
        )

        p1_tm = float(self.p1_team_rate.get(tok["p1_sig"], g))
        p2_tm = float(self.p2_team_rate.get(tok["p2_sig"], 1.0 - self.p1_team_rate.get(tok["p2_sig"], g)))

        return {
            "te_p1_species_winrate_mean": p1_sp_mean,
            "te_p1_species_winrate_min": float(np.min(p1_rates)) if p1_rates else g,
            "te_p2_species_winrate_mean": p2_sp_mean,
            "te_species_winrate_diff": p1_sp_mean - (1.0 - p2_sp_mean),
            "te_p1_alive_species_winrate_mean": p1_alv_mean,
            "te_p2_alive_species_winrate_mean": p2_alv_mean,
            "te_alive_species_winrate_diff": p1_alv_mean - (1.0 - p2_alv_mean),
            "te_p1_hp_wt_winrate": float(p1_hp_wt),
            "te_p2_hp_wt_winrate": float(p2_hp_wt),
            "te_hp_wt_winrate_diff": float(p1_hp_wt - (1.0 - p2_hp_wt)),
            "te_p1_team_winrate": p1_tm,
            "te_p2_team_winrate": p2_tm,
        }

    def transform(self, battles: Sequence[Dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame([self.transform_single(b) for b in battles])

    def fit_transform_inner_oof(
        self,
        train_battles: Sequence[Dict[str, Any]],
        y_train: np.ndarray,
        n_inner_splits: int = 5,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Compute inner out-of-fold encoded features on `train_battles` to prevent
        overfitting on the training fold, then fit on all `train_battles` for val/test transform.
        """
        from sklearn.model_selection import StratifiedKFold

        inner_skf = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=seed)
        oof_rows: List[Optional[Dict[str, float]]] = [None] * len(train_battles)
        for in_tr, in_va in inner_skf.split(np.zeros(len(y_train)), y_train):
            sub_enc = WithinFoldTargetEncoder(smoothing=self.smoothing, team_smoothing=self.team_smoothing)
            sub_enc.fit([train_battles[i] for i in in_tr], y_train[in_tr])
            for idx in in_va:
                oof_rows[idx] = sub_enc.transform_single(train_battles[idx])

        self.fit(train_battles, y_train)
        return pd.DataFrame(oof_rows)
