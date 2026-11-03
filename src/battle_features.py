"""Core battle state reconstruction, Gen 2 mechanics, and per-battle feature families."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.data_loader import get_type_effectiveness


# Key Generation 2 OU species (top meta staples observed in the dataset)
TOP_META_SPECIES: Tuple[str, ...] = (
    "snorlax",
    "cloyster",
    "zapdos",
    "raikou",
    "tyranitar",
    "exeggutor",
    "gengar",
    "skarmory",
    "steelix",
    "machamp",
    "forretress",
    "starmie",
    "nidoking",
    "golem",
    "marowak",
    "umbreon",
    "vaporeon",
    "miltank",
    "jynx",
    "suicune",
    "blissey",
    "misdreavus",
    "heracross",
    "jolteon",
    "rhydon",
)

RECOVERY_MOVES: Set[str] = {
    "rest",
    "recover",
    "softboiled",
    "milkdrink",
    "moonlight",
    "synthesis",
    "morningsun",
    "painsplit",
    "leechseed",
}

DIRECT_HEAL_MOVES: Set[str] = {
    "rest",
    "recover",
    "softboiled",
    "milkdrink",
    "moonlight",
    "synthesis",
    "morningsun",
}

PHAZING_MOVES: Set[str] = {"roar", "whirlwind"}

SETUP_MOVES: Set[str] = {
    "curse",
    "bellydrum",
    "swordsdance",
    "growth",
    "agility",
    "amnesia",
    "barrier",
    "meditate",
    "sharpen",
    "minimize",
    "doubleteam",
}

SELF_KO_MOVES: Set[str] = {"explosion", "selfdestruct", "destinybond", "perishsong"}

SLEEP_INDUCING_MOVES: Set[str] = {"sleeppowder", "lovelykiss", "hypnosis", "spore", "sing"}

PARA_INDUCING_MOVES: Set[str] = {"thunderwave", "stunspore", "glare", "bodyslam", "zapcannon"}

POISON_INDUCING_MOVES: Set[str] = {"toxic", "poisonpowder", "sludgebomb"}

FIXED_DAMAGE_MOVES: Set[str] = {
    "seismictoss",
    "nightshade",
    "counter",
    "mirrorcoat",
    "superfang",
    "fissure",
    "guillotine",
    "horndrill",
}

RECOIL_MOVES: Set[str] = {"doubleedge", "submission", "takedown"}


def get_stage_multiplier(stage: int) -> float:
    """Gen 2 stat stage multiplier for stages in [-6, +6]."""
    s = max(-6, min(6, int(stage)))
    if s >= 0:
        return (2.0 + s) / 2.0
    return 2.0 / (2.0 - s)


def build_species_catalogue(battles: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build a lookup table mapping species `name` -> base stats & types strictly from
    `p1_team_details` and `p2_lead_details` of the provided battles (e.g. training fold).
    """
    catalogue: Dict[str, Dict[str, Any]] = {}
    for b in battles:
        for p in b.get("p1_team_details", []):
            name = p.get("name")
            if name and name not in catalogue:
                catalogue[name] = {
                    "types": [t for t in p.get("types", []) if t and t != "notype"],
                    "base_hp": float(p.get("base_hp", 80)),
                    "base_atk": float(p.get("base_atk", 80)),
                    "base_def": float(p.get("base_def", 80)),
                    "base_spa": float(p.get("base_spa", 80)),
                    "base_spd": float(p.get("base_spd", 80)),
                    "base_spe": float(p.get("base_spe", 80)),
                }
        lead = b.get("p2_lead_details")
        if lead:
            name = lead.get("name")
            if name and name not in catalogue:
                catalogue[name] = {
                    "types": [t for t in lead.get("types", []) if t and t != "notype"],
                    "base_hp": float(lead.get("base_hp", 80)),
                    "base_atk": float(lead.get("base_atk", 80)),
                    "base_def": float(lead.get("base_def", 80)),
                    "base_spa": float(lead.get("base_spa", 80)),
                    "base_spd": float(lead.get("base_spd", 80)),
                    "base_spe": float(lead.get("base_spe", 80)),
                }
    return catalogue


def get_species_stats(
    name: str,
    catalogue: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Retrieve species stats from catalogue with safe fallback for unseen species."""
    if name in catalogue:
        return catalogue[name]
    return {
        "types": [],
        "base_hp": 80.0,
        "base_atk": 80.0,
        "base_def": 80.0,
        "base_spa": 80.0,
        "base_spd": 80.0,
        "base_spe": 80.0,
    }


def extract_organizer_features(battle: Dict[str, Any]) -> Dict[str, float]:
    """Exact 6 features from the official starter notebook (`notebook baseline.ipynb`)."""
    feats: Dict[str, float] = {}
    p1_team = battle.get("p1_team_details", [])
    if p1_team:
        feats["p1_mean_hp"] = float(np.mean([p.get("base_hp", 0) for p in p1_team]))
        feats["p1_mean_spe"] = float(np.mean([p.get("base_spe", 0) for p in p1_team]))
        feats["p1_mean_atk"] = float(np.mean([p.get("base_atk", 0) for p in p1_team]))
        feats["p1_mean_def"] = float(np.mean([p.get("base_def", 0) for p in p1_team]))
    else:
        feats["p1_mean_hp"] = 0.0
        feats["p1_mean_spe"] = 0.0
        feats["p1_mean_atk"] = 0.0
        feats["p1_mean_def"] = 0.0

    p2_lead = battle.get("p2_lead_details") or {}
    feats["p2_lead_hp"] = float(p2_lead.get("base_hp", 0))
    feats["p2_lead_spe"] = float(p2_lead.get("base_spe", 0))
    return feats


def extract_team_features(battle: Dict[str, Any]) -> Dict[str, float]:
    """Family A: Player 1 full 6-Pokémon team composition & Player 2 lead characteristics."""
    feats: Dict[str, float] = {}
    p1_team = battle.get("p1_team_details", [])
    p2_lead = battle.get("p2_lead_details") or {}

    stat_Names = ("hp", "atk", "def", "spa", "spd", "spe")
    stat_arrays: Dict[str, np.ndarray] = {}
    for st in stat_Names:
        arr = np.array([float(p.get(f"base_{st}", 0.0)) for p in p1_team], dtype=float)
        if len(arr) == 0:
            arr = np.zeros(1, dtype=float)
        stat_arrays[st] = arr
        feats[f"p1_mean_{st}"] = float(np.mean(arr))
        feats[f"p1_sum_{st}"] = float(np.sum(arr))
        feats[f"p1_min_{st}"] = float(np.min(arr))
        feats[f"p1_max_{st}"] = float(np.max(arr))
        feats[f"p1_std_{st}"] = float(np.std(arr))

    bst_arr = sum(stat_arrays[st] for st in stat_Names)
    feats["p1_mean_bst"] = float(np.mean(bst_arr))
    feats["p1_sum_bst"] = float(np.sum(bst_arr))
    feats["p1_min_bst"] = float(np.min(bst_arr))
    feats["p1_max_bst"] = float(np.max(bst_arr))
    feats["p1_std_bst"] = float(np.std(bst_arr))

    # Offensive vs defensive balance
    off_arr = stat_arrays["atk"] + stat_arrays["spa"]
    def_arr = stat_arrays["def"] + stat_arrays["spd"] + stat_arrays["hp"]
    best_off_arr = np.maximum(stat_arrays["atk"], stat_arrays["spa"])
    feats["p1_mean_best_off"] = float(np.mean(best_off_arr))
    feats["p1_max_best_off"] = float(np.max(best_off_arr))
    feats["p1_off_def_ratio"] = float(np.sum(off_arr) / max(1.0, np.sum(def_arr)))
    feats["p1_phys_spec_ratio"] = float(np.sum(stat_arrays["atk"]) / max(1.0, np.sum(stat_arrays["spa"])))

    # Speed & bulk tiers
    feats["p1_cnt_spe_ge_100"] = float(np.sum(stat_arrays["spe"] >= 100))
    feats["p1_cnt_spe_ge_115"] = float(np.sum(stat_arrays["spe"] >= 115))
    feats["p1_cnt_spe_le_60"] = float(np.sum(stat_arrays["spe"] <= 60))
    feats["p1_cnt_hp_ge_100"] = float(np.sum(stat_arrays["hp"] >= 100))
    feats["p1_cnt_def_ge_110"] = float(np.sum(stat_arrays["def"] >= 110))
    feats["p1_cnt_spd_ge_100"] = float(np.sum(stat_arrays["spd"] >= 100))

    # Level features
    levels = [float(p.get("level", 100)) for p in p1_team]
    feats["p1_mean_level"] = float(np.mean(levels)) if levels else 100.0
    feats["p1_is_all_lv100"] = 1.0 if all(lv == 100 for lv in levels) else 0.0

    # Held items
    items = [str(p.get("item", "")).lower() for p in p1_team]
    feats["p1_leftovers_cnt"] = float(sum(1 for it in items if it == "leftovers"))
    feats["p1_berry_cnt"] = float(sum(1 for it in items if "berry" in it or it == "berryjuice"))
    feats["p1_thickclub_cnt"] = float(sum(1 for it in items if it == "thickclub"))
    feats["p1_unique_items_cnt"] = float(len(set(items)))

    # Type diversity & coverage on P1 team
    p1_types_all: List[str] = []
    for p in p1_team:
        for t in p.get("types", []):
            if t and t != "notype":
                p1_types_all.append(t.lower())
    unique_p1_types = set(p1_types_all)
    feats["p1_unique_types_cnt"] = float(len(unique_p1_types))
    feats["p1_dual_type_cnt"] = float(
        sum(1 for p in p1_team if sum(1 for t in p.get("types", []) if t and t != "notype") == 2)
    )

    for key_t in ("normal", "electric", "ground", "rock", "steel", "water", "psychic", "ghost", "flying", "fighting", "grass", "ice", "fire", "dark", "poison", "bug", "dragon"):
        feats[f"p1_type_cnt_{key_t}"] = float(sum(1 for t in p1_types_all if t == key_t))

    # P2 lead stats
    p2_lead_bst = 0.0
    for st in stat_Names:
        val = float(p2_lead.get(f"base_{st}", 0.0))
        feats[f"p2_lead_{st}"] = val
        p2_lead_bst += val
    feats["p2_lead_bst"] = p2_lead_bst
    feats["p2_lead_best_off"] = max(feats["p2_lead_atk"], feats["p2_lead_spa"])
    feats["p2_lead_bulk"] = feats["p2_lead_hp"] + feats["p2_lead_def"] + feats["p2_lead_spd"]
    feats["p2_lead_level"] = float(p2_lead.get("level", 100))

    # Relative P1 team vs P2 lead
    feats["p1_vs_p2_lead_bst_diff"] = feats["p1_mean_bst"] - p2_lead_bst
    feats["p1_vs_p2_lead_spe_diff"] = feats["p1_mean_spe"] - feats["p2_lead_spe"]
    feats["p1_cnt_outspeed_p2_lead"] = float(np.sum(stat_arrays["spe"] > feats["p2_lead_spe"]))

    # P1 Lead (first Pokémon seen on Turn 1) vs P2 Lead
    tl = battle.get("battle_timeline", [])
    p1_lead_name = tl[0]["p1_pokemon_state"]["name"] if tl else (p1_team[0]["name"] if p1_team else "")
    p1_lead_dict = next((p for p in p1_team if p.get("name") == p1_lead_name), p1_team[0] if p1_team else {})
    p1_lead_spe = float(p1_lead_dict.get("base_spe", 80.0))
    p1_lead_bst = sum(float(p1_lead_dict.get(f"base_{st}", 80.0)) for st in stat_Names)
    feats["p1_lead_spe"] = p1_lead_spe
    feats["p1_lead_bst"] = p1_lead_bst
    feats["lead_spe_diff"] = p1_lead_spe - feats["p2_lead_spe"]
    feats["lead_outspeeds_p2"] = 1.0 if p1_lead_spe > feats["p2_lead_spe"] else (0.5 if p1_lead_spe == feats["p2_lead_spe"] else 0.0)
    feats["lead_bst_diff"] = p1_lead_bst - p2_lead_bst

    return feats


def extract_simple_state_features(battle: Dict[str, Any]) -> Dict[str, float]:
    """Minimal battle-state aggregates for the Initial XGBoost Baseline (Section 8)."""
    feats: Dict[str, float] = {}
    tl = battle.get("battle_timeline", [])
    p1_team = battle.get("p1_team_details", [])
    p2_lead = battle.get("p2_lead_details") or {}

    p1_hp = {p.get("name", f"u_{i}"): 1.0 for i, p in enumerate(p1_team)}
    p2_hp = {p2_lead.get("name", "lead"): 1.0} if p2_lead else {}

    for e in tl:
        s1 = e.get("p1_pokemon_state") or {}
        s2 = e.get("p2_pokemon_state") or {}
        n1 = s1.get("name")
        n2 = s2.get("name")
        if n1:
            p1_hp[n1] = 0.0 if s1.get("status") == "fnt" else float(s1.get("hp_pct", 1.0))
        if n2:
            p2_hp[n2] = 0.0 if s2.get("status") == "fnt" else float(s2.get("hp_pct", 1.0))

    last_e = tl[-1] if tl else {}
    last_s1 = last_e.get("p1_pokemon_state") or {}
    last_s2 = last_e.get("p2_pokemon_state") or {}

    feats["simple_p1_active_hp"] = 0.0 if last_s1.get("status") == "fnt" else float(last_s1.get("hp_pct", 1.0))
    feats["simple_p2_active_hp"] = 0.0 if last_s2.get("status") == "fnt" else float(last_s2.get("hp_pct", 1.0))
    feats["simple_p1_fainted_cnt"] = float(sum(1 for v in p1_hp.values() if v <= 0.0))
    feats["simple_p2_fainted_exp_cnt"] = float(sum(1 for v in p2_hp.values() if v <= 0.0))
    feats["simple_p2_revealed_cnt"] = float(len(p2_hp))

    n_unrev_p2 = max(0, 6 - len(p2_hp))
    p1_total_hp = float(sum(p1_hp.values()))
    p2_total_hp = float(sum(p2_hp.values()) + 1.0 * n_unrev_p2)
    feats["simple_p1_total_hp"] = p1_total_hp
    feats["simple_p2_total_hp_raw"] = p2_total_hp
    feats["simple_hp_diff_raw"] = p1_total_hp - p2_total_hp
    return feats


def reconstruct_battle_history(
    battle: Dict[str, Any],
    catalogue: Dict[str, Dict[str, Any]],
    type_chart: Dict[Tuple[str, str], float],
) -> Dict[str, Any]:
    """Single-pass reconstruction of the 30-turn battle timeline without double-counting cached moves."""
    tl = battle.get("battle_timeline", [])
    p1_team = battle.get("p1_team_details", [])
    p2_lead = battle.get("p2_lead_details") or {}

    p1_team_map: Dict[str, Dict[str, Any]] = {}
    for p in p1_team:
        name = p.get("name", "")
        p1_team_map[name] = {
            "types": [t for t in p.get("types", []) if t and t != "notype"],
            "base_hp": float(p.get("base_hp", 80)),
            "base_atk": float(p.get("base_atk", 80)),
            "base_def": float(p.get("base_def", 80)),
            "base_spa": float(p.get("base_spa", 80)),
            "base_spd": float(p.get("base_spd", 80)),
            "base_spe": float(p.get("base_spe", 80)),
            "item": str(p.get("item", "")).lower(),
            "level": float(p.get("level", 100)),
        }

    # Find last turn each Pokémon appears
    last_seen_p1: Dict[str, int] = {}
    last_seen_p2: Dict[str, int] = {}
    for i, e in enumerate(tl):
        last_seen_p1[e["p1_pokemon_state"]["name"]] = i
        last_seen_p2[e["p2_pokemon_state"]["name"]] = i

    # State trackers across turns
    p1_hp: Dict[str, float] = {p["name"]: 1.0 for p in p1_team}
    p1_status: Dict[str, str] = {p["name"]: "nostatus" for p in p1_team}
    p1_seen: Set[str] = set()
    p1_moves_by_mon: Dict[str, Set[str]] = {p["name"]: set() for p in p1_team}
    p1_move_objs_by_mon: Dict[str, Dict[str, Dict[str, Any]]] = {p["name"]: {} for p in p1_team}

    p2_lead_name = p2_lead.get("name", "")
    p2_hp: Dict[str, float] = {p2_lead_name: 1.0} if p2_lead_name else {}
    p2_status: Dict[str, str] = {p2_lead_name: "nostatus"} if p2_lead_name else {}
    p2_moves_by_mon: Dict[str, Set[str]] = {p2_lead_name: set()} if p2_lead_name else {}
    p2_move_objs_by_mon: Dict[str, Dict[str, Dict[str, Any]]] = {p2_lead_name: {}} if p2_lead_name else {}

    p2_explicit_faints: Set[str] = set()
    p2_inferred_kos_strict: Set[str] = set()
    p2_ko_probs: Dict[str, float] = {}

    last_cached_move_p1: Dict[str, str] = {}
    last_cached_move_p2: Dict[str, str] = {}
    prev_hp_p1: Dict[str, float] = dict(p1_hp)
    prev_hp_p2: Dict[str, float] = dict(p2_hp)
    prev_st_p2: Dict[str, str] = dict(p2_status)

    # Turn-level arrays (length 30)
    n_turns = len(tl)
    h1_traj = np.zeros(n_turns, dtype=float)
    h2_raw_traj = np.zeros(n_turns, dtype=float)
    h2_adj_traj = np.zeros(n_turns, dtype=float)
    f1_traj = np.zeros(n_turns, dtype=float)
    f2_exp_traj = np.zeros(n_turns, dtype=float)
    f2_adj_traj = np.zeros(n_turns, dtype=float)
    act_hp_diff_traj = np.zeros(n_turns, dtype=float)
    boost_diff_traj = np.zeros(n_turns, dtype=float)

    p1_switched_turn = np.zeros(n_turns, dtype=float)
    p2_switched_turn = np.zeros(n_turns, dtype=float)
    p1_attacked_turn = np.zeros(n_turns, dtype=float)
    p2_attacked_turn = np.zeros(n_turns, dtype=float)
    p1_move_bp_turn = np.zeros(n_turns, dtype=float)
    p2_move_bp_turn = np.zeros(n_turns, dtype=float)
    p1_dmg_taken_turn = np.zeros(n_turns, dtype=float)
    p2_dmg_taken_turn = np.zeros(n_turns, dtype=float)
    p1_healed_turn = np.zeros(n_turns, dtype=float)
    p2_healed_turn = np.zeros(n_turns, dtype=float)

    p1_spikes_used = 0
    p2_spikes_used = 0
    p1_rapidspin_used = 0
    p2_rapidspin_used = 0
    p1_phaze_used = 0
    p2_phaze_used = 0
    p1_Explosion_used = 0
    p2_explosion_used = 0

    for i, e in enumerate(tl):
        s1 = e["p1_pokemon_state"]
        s2 = e["p2_pokemon_state"]
        m1 = e.get("p1_move_details")
        m2 = e.get("p2_move_details")
        n1 = s1["name"]
        n2 = s2["name"]

        p1_seen.add(n1)
        if n2 not in p2_hp:
            p2_hp[n2] = 1.0
            p2_status[n2] = "nostatus"
            p2_moves_by_mon[n2] = set()
            p2_move_objs_by_mon[n2] = {}

        cur_hp1 = 0.0 if s1["status"] == "fnt" else float(s1["hp_pct"])
        cur_hp2 = 0.0 if s2["status"] == "fnt" else float(s2["hp_pct"])

        # Determine if P1 or P2 switched on turn i vs executed a move
        p1_name_changed = (i > 0 and tl[i - 1]["p1_pokemon_state"]["name"] != n1)
        p2_name_changed = (i > 0 and tl[i - 1]["p2_pokemon_state"]["name"] != n2)
        p1_prev_fnt = (i > 0 and tl[i - 1]["p1_pokemon_state"]["status"] == "fnt")
        p2_prev_fnt = (i > 0 and tl[i - 1]["p2_pokemon_state"]["status"] == "fnt")

        p1_vol_switch = p1_name_changed and not p1_prev_fnt
        p2_vol_switch = p2_name_changed and not p2_prev_fnt

        if p1_vol_switch:
            p1_switched_turn[i] = 1.0
        if p2_vol_switch:
            p2_switched_turn[i] = 1.0

        # Did P1 actually execute m1 on turn i?
        p1_executed = False
        if m1 is not None:
            m1_name = m1["name"]
            p1_moves_by_mon.setdefault(n1, set()).add(m1_name)
            p1_move_objs_by_mon.setdefault(n1, {})[m1_name] = m1
            if not p1_vol_switch:
                p1_executed = True
            elif m1_name != last_cached_move_p1.get(n1):
                # Even if n1 changed at end of turn (e.g. phazed), if m1 is new, a move occurred
                p1_executed = True
            last_cached_move_p1[n1] = m1_name

        # Did P2 actually execute m2 on turn i?
        p2_executed = False
        if m2 is not None:
            m2_name = m2["name"]
            p2_moves_by_mon.setdefault(n2, set()).add(m2_name)
            p2_move_objs_by_mon.setdefault(n2, {})[m2_name] = m2
            if not p2_vol_switch:
                p2_executed = True
            elif m2_name != last_cached_move_p2.get(n2):
                p2_executed = True
            last_cached_move_p2[n2] = m2_name

        if p1_executed and m1 is not None:
            m1_name = m1["name"]
            bp1 = float(m1.get("base_power", 0))
            if bp1 > 0 or m1_name in FIXED_DAMAGE_MOVES:
                p1_attacked_turn[i] = 1.0
                p1_move_bp_turn[i] = bp1 if bp1 > 0 else 75.0
            if m1_name == "spikes":
                p1_spikes_used += 1
            elif m1_name == "rapidspin":
                p1_rapidspin_used += 1
                p2_spikes_used = 0
            elif m1_name in PHAZING_MOVES:
                p1_phaze_used += 1
            elif m1_name in {"explosion", "selfdestruct"}:
                p1_Explosion_used += 1

        if p2_executed and m2 is not None:
            m2_name = m2["name"]
            bp2 = float(m2.get("base_power", 0))
            if bp2 > 0 or m2_name in FIXED_DAMAGE_MOVES:
                p2_attacked_turn[i] = 1.0
                p2_move_bp_turn[i] = bp2 if bp2 > 0 else 75.0
            if m2_name == "spikes":
                p2_spikes_used += 1
            elif m2_name == "rapidspin":
                p2_rapidspin_used += 1
                p1_spikes_used = 0
            elif m2_name in PHAZING_MOVES:
                p2_phaze_used += 1
            elif m2_name in {"explosion", "selfdestruct"}:
                p2_explosion_used += 1

        # Track single-turn HP changes for n1 and n2 relative to their previous known HP
        old_hp1 = prev_hp_p1.get(n1, 1.0)
        if cur_hp1 < old_hp1:
            p1_dmg_taken_turn[i] = old_hp1 - cur_hp1
        elif cur_hp1 > old_hp1:
            p1_healed_turn[i] = cur_hp1 - old_hp1

        old_hp2 = prev_hp_p2.get(n2, 1.0)
        if cur_hp2 < old_hp2:
            p2_dmg_taken_turn[i] = old_hp2 - cur_hp2
        elif cur_hp2 > old_hp2:
            p2_healed_turn[i] = cur_hp2 - old_hp2

        # Physics-based P2 KO detection
        if s2["status"] == "fnt" or cur_hp2 <= 0.0:
            p2_explicit_faints.add(n2)
            p2_inferred_kos_strict.add(n2)
            p2_ko_probs[n2] = 1.0

        if i > 0:
            s1_prev = tl[i - 1]["p1_pokemon_state"]
            s2_prev = tl[i - 1]["p2_pokemon_state"]
            n2_prev = s2_prev["name"]

            if (
                n2 != n2_prev
                and s2_prev["status"] != "fnt"
                and float(s2_prev["hp_pct"]) > 0.0
                and last_seen_p2.get(n2_prev, -1) == i - 1
            ):
                # n2_prev disappeared permanently after turn i-1!
                exp_y_hp = prev_hp_p2.get(n2, 1.0)
                y_untouched = (cur_hp2 >= exp_y_hp - 0.13) and (s2["status"] == prev_st_p2.get(n2, "nostatus"))

                m1_name = m1["name"] if m1 else ""
                m1_bp = float(m1.get("base_power", 0)) if m1 else 0.0
                m1_type = m1.get("type", "notype") if m1 else "notype"
                prev_x_hp = float(s2_prev["hp_pct"])

                # Check type effectiveness of P1's move against n2_prev and n2
                x_types = get_species_stats(n2_prev, catalogue)["types"]
                y_types = get_species_stats(n2, catalogue)["types"]
                eff_vs_x = get_type_effectiveness(m1_type, x_types, type_chart) if m1_bp > 0 else 1.0
                eff_vs_y = get_type_effectiveness(m1_type, y_types, type_chart) if m1_bp > 0 else 1.0

                m1_is_attack = (
                    p1_executed
                    and m1_name not in PHAZING_MOVES
                    and (m1_bp > 0 or m1_name in FIXED_DAMAGE_MOVES)
                )

                # Did P1 take recoil or heavy damage (e.g. from n2_prev's Explosion)?
                p1_prev_known_hp = prev_hp_p1.get(n1, 1.0)
                p1_hp_drop = p1_prev_known_hp - cur_hp1

                residual_death = (
                    (s2_prev["status"] in {"tox", "psn", "brn"} or any(ef in s2_prev.get("effects", []) for ef in ("leechseed", "perish", "curse", "nightmare")))
                    and prev_x_hp <= 0.28
                )

                if y_untouched:
                    if m1_is_attack and eff_vs_x > 0:
                        # Estimate if prev_x_hp was in KO range or if y would have taken damage had y switched in at start of turn
                        if prev_x_hp <= 0.55 or (prev_x_hp <= 0.80 and eff_vs_x >= 2.0):
                            p2_inferred_kos_strict.add(n2_prev)
                            p2_ko_probs[n2_prev] = max(p2_ko_probs.get(n2_prev, 0.0), 0.90)
                        elif eff_vs_y >= 1.0 and cur_hp2 >= exp_y_hp - 0.01 and float(m1.get("accuracy", 1.0)) >= 0.95:
                            # If y switched in manually, a 95-100% accurate neutral/SE move would have damaged y!
                            p2_inferred_kos_strict.add(n2_prev)
                            p2_ko_probs[n2_prev] = max(p2_ko_probs.get(n2_prev, 0.0), 0.85)
                        else:
                            p2_ko_probs[n2_prev] = max(p2_ko_probs.get(n2_prev, 0.0), 0.55)
                    elif p1_hp_drop >= 0.22 and cur_hp2 >= 0.87:
                        # n2_prev likely exploded on P1!
                        p2_inferred_kos_strict.add(n2_prev)
                        p2_ko_probs[n2_prev] = max(p2_ko_probs.get(n2_prev, 0.0), 0.88)
                    elif residual_death:
                        p2_inferred_kos_strict.add(n2_prev)
                        p2_ko_probs[n2_prev] = max(p2_ko_probs.get(n2_prev, 0.0), 0.85)
                    elif prev_x_hp <= 0.20:
                        p2_ko_probs[n2_prev] = max(p2_ko_probs.get(n2_prev, 0.0), 0.65)
                else:
                    if prev_x_hp <= 0.15:
                        p2_ko_probs[n2_prev] = max(p2_ko_probs.get(n2_prev, 0.0), 0.35)

        # Update current state maps
        p1_hp[n1] = cur_hp1
        p1_status[n1] = s1["status"]
        p2_hp[n2] = cur_hp2
        p2_status[n2] = s2["status"]
        prev_hp_p1[n1] = cur_hp1
        prev_hp_p2[n2] = cur_hp2
        prev_st_p2[n2] = s2["status"]

        # Record turn trajectories
        n_unrev_p2 = max(0, 6 - len(p2_hp))
        h1_traj[i] = sum(p1_hp.values())
        h2_raw_traj[i] = sum(p2_hp.values()) + 1.0 * n_unrev_p2
        h2_adj_traj[i] = sum(
            0.0 if k in p2_inferred_kos_strict else v for k, v in p2_hp.items()
        ) + 1.0 * n_unrev_p2

        f1_traj[i] = sum(1.0 for v in p1_hp.values() if v <= 0.0)
        f2_exp_traj[i] = float(len(p2_explicit_faints))
        f2_adj_traj[i] = float(len(p2_inferred_kos_strict))
        act_hp_diff_traj[i] = cur_hp1 - cur_hp2

        b1 = sum(int(v) for v in (s1.get("boosts") or {}).values())
        b2 = sum(int(v) for v in (s2.get("boosts") or {}).values())
        boost_diff_traj[i] = float(b1 - b2)

    return {
        "p1_team_map": p1_team_map,
        "p1_hp": p1_hp,
        "p1_status": p1_status,
        "p1_seen": p1_seen,
        "p1_moves_by_mon": p1_moves_by_mon,
        "p1_move_objs_by_mon": p1_move_objs_by_mon,
        "p2_hp": p2_hp,
        "p2_status": p2_status,
        "p2_moves_by_mon": p2_moves_by_mon,
        "p2_move_objs_by_mon": p2_move_objs_by_mon,
        "p2_explicit_faints": p2_explicit_faints,
        "p2_inferred_kos_strict": p2_inferred_kos_strict,
        "p2_ko_probs": p2_ko_probs,
        "last_seen_p1": last_seen_p1,
        "last_seen_p2": last_seen_p2,
        "h1_traj": h1_traj,
        "h2_raw_traj": h2_raw_traj,
        "h2_adj_traj": h2_adj_traj,
        "f1_traj": f1_traj,
        "f2_exp_traj": f2_exp_traj,
        "f2_adj_traj": f2_adj_traj,
        "act_hp_diff_traj": act_hp_diff_traj,
        "boost_diff_traj": boost_diff_traj,
        "p1_switched_turn": p1_switched_turn,
        "p2_switched_turn": p2_switched_turn,
        "p1_attacked_turn": p1_attacked_turn,
        "p2_attacked_turn": p2_attacked_turn,
        "p1_move_bp_turn": p1_move_bp_turn,
        "p2_move_bp_turn": p2_move_bp_turn,
        "p1_dmg_taken_turn": p1_dmg_taken_turn,
        "p2_dmg_taken_turn": p2_dmg_taken_turn,
        "p1_healed_turn": p1_healed_turn,
        "p2_healed_turn": p2_healed_turn,
        "p1_spikes_used": p1_spikes_used,
        "p2_spikes_used": p2_spikes_used,
        "p1_rapidspin_used": p1_rapidspin_used,
        "p2_rapidspin_used": p2_rapidspin_used,
        "p1_phaze_used": p1_phaze_used,
        "p2_phaze_used": p2_phaze_used,
        "p1_explosion_used": p1_Explosion_used,
        "p2_explosion_used": p2_explosion_used,
    }


def extract_p2_observed_features(
    battle: Dict[str, Any],
    recon: Dict[str, Any],
    catalogue: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    """Family B: Player 2 observed team, revealed species, moves, and unrevealed backline."""
    feats: Dict[str, float] = {}
    p1_team_map = recon["p1_team_map"]
    p2_hp = recon["p2_hp"]
    p1_seen = recon["p1_seen"]
    p1_moves_by_mon = recon["p1_moves_by_mon"]
    p2_moves_by_mon = recon["p2_moves_by_mon"]
    p1_move_objs = recon["p1_move_objs_by_mon"]
    p2_move_objs = recon["p2_move_objs_by_mon"]

    n_rev_p2 = len(p2_hp)
    n_unrev_p2 = max(0, 6 - n_rev_p2)
    n_seen_p1 = len(p1_seen)
    n_unseen_p1 = max(0, 6 - n_seen_p1)

    feats["p2_revealed_cnt"] = float(n_rev_p2)
    feats["p2_unrevealed_cnt"] = float(n_unrev_p2)
    feats["p1_seen_cnt"] = float(n_seen_p1)
    feats["p1_unseen_cnt"] = float(n_unseen_p1)
    feats["revealed_cnt_diff"] = float(n_rev_p2 - n_seen_p1)

    # Stats of revealed P2 Pokémon (from within-fold catalogue)
    stat_names = ("hp", "atk", "def", "spa", "spd", "spe")
    p2_rev_stats: Dict[str, List[float]] = {st: [] for st in stat_names}
    p2_rev_bst: List[float] = []
    p2_rev_types: List[str] = []

    for mon_name in p2_hp.keys():
        sp = get_species_stats(mon_name, catalogue)
        bst = 0.0
        for st in stat_names:
            val = float(sp[f"base_{st}"])
            p2_rev_stats[st].append(val)
            bst += val
        p2_rev_bst.append(bst)
        p2_rev_types.extend(sp["types"])

    for st in stat_names:
        arr = np.array(p2_rev_stats[st], dtype=float) if p2_rev_stats[st] else np.array([80.0])
        feats[f"p2_rev_mean_{st}"] = float(np.mean(arr))
        feats[f"p2_rev_max_{st}"] = float(np.max(arr))
        feats[f"p2_rev_min_{st}"] = float(np.min(arr))

    bst_arr = np.array(p2_rev_bst, dtype=float) if p2_rev_bst else np.array([480.0])
    feats["p2_rev_mean_bst"] = float(np.mean(bst_arr))
    feats["p2_rev_max_bst"] = float(np.max(bst_arr))
    feats["p2_rev_min_bst"] = float(np.min(bst_arr))
    feats["p2_rev_unique_types_cnt"] = float(len(set(p2_rev_types)))
    feats["p2_rev_cnt_spe_ge_100"] = float(sum(1 for v in p2_rev_stats["spe"] if v >= 100))

    # Observed move repertoire for P1 and P2
    p1_all_moves: Set[str] = set()
    p2_all_moves: Set[str] = set()
    p1_all_move_objs: List[Dict[str, Any]] = []
    p2_all_move_objs: List[Dict[str, Any]] = []

    for mset in p1_moves_by_mon.values():
        p1_all_moves.update(mset)
    for mset in p2_moves_by_mon.values():
        p2_all_moves.update(mset)
    for mdict in p1_move_objs.values():
        p1_all_move_objs.extend(mdict.values())
    for mdict in p2_move_objs.values():
        p2_all_move_objs.extend(mdict.values())

    feats["p1_total_revealed_moves"] = float(sum(len(s) for s in p1_moves_by_mon.values()))
    feats["p2_total_revealed_moves"] = float(sum(len(s) for s in p2_moves_by_mon.values()))
    feats["p1_moves_per_seen_mon"] = feats["p1_total_revealed_moves"] / max(1.0, float(n_seen_p1))
    feats["p2_moves_per_rev_mon"] = feats["p2_total_revealed_moves"] / max(1.0, float(n_rev_p2))

    # Move category & power summaries
    for side, mobjs, mnames in [("p1", p1_all_move_objs, p1_all_moves), ("p2", p2_all_move_objs, p2_all_moves)]:
        phys_cnt = sum(1 for m in mobjs if m.get("category") == "physical")
        spec_cnt = sum(1 for m in mobjs if m.get("category") == "special")
        stat_cnt = sum(1 for m in mobjs if m.get("category") == "status")
        bps = [float(m.get("base_power", 0)) for m in mobjs if float(m.get("base_power", 0)) > 0]
        feats[f"{side}_obs_phys_moves_cnt"] = float(phys_cnt)
        feats[f"{side}_obs_spec_moves_cnt"] = float(spec_cnt)
        feats[f"{side}_obs_stat_moves_cnt"] = float(stat_cnt)
        feats[f"{side}_obs_max_bp"] = float(max(bps)) if bps else 0.0
        feats[f"{side}_obs_mean_bp"] = float(np.mean(bps)) if bps else 0.0

        # Strategic Gen 2 move indicators
        feats[f"{side}_has_recovery_move"] = float(len(mnames & RECOVERY_MOVES))
        feats[f"{side}_has_direct_heal"] = float(len(mnames & DIRECT_HEAL_MOVES))
        feats[f"{side}_has_rest"] = 1.0 if "rest" in mnames else 0.0
        feats[f"{side}_has_sleeptalk"] = 1.0 if "sleeptalk" in mnames else 0.0
        feats[f"{side}_has_healbell"] = 1.0 if "healbell" in mnames else 0.0
        feats[f"{side}_has_spikes"] = 1.0 if "spikes" in mnames else 0.0
        feats[f"{side}_has_rapidspin"] = 1.0 if "rapidspin" in mnames else 0.0
        feats[f"{side}_has_phaze"] = float(len(mnames & PHAZING_MOVES))
        feats[f"{side}_has_setup"] = float(len(mnames & SETUP_MOVES))
        feats[f"{side}_has_curse"] = 1.0 if "curse" in mnames else 0.0
        feats[f"{side}_has_sleep_move"] = float(len(mnames & SLEEP_INDUCING_MOVES))
        feats[f"{side}_has_para_move"] = float(len(mnames & PARA_INDUCING_MOVES))
        feats[f"{side}_has_poison_move"] = float(len(mnames & POISON_INDUCING_MOVES))
        feats[f"{side}_has_self_ko"] = float(len(mnames & SELF_KO_MOVES))

    # Count of Pokémon with Rest + SleepTalk (RestTalkers)
    feats["p1_resttalk_mon_cnt"] = float(
        sum(1 for mset in p1_moves_by_mon.values() if "rest" in mset and "sleeptalk" in mset)
    )
    feats["p2_resttalk_mon_cnt"] = float(
        sum(1 for mset in p2_moves_by_mon.values() if "rest" in mset and "sleeptalk" in mset)
    )

    # Top Meta Species presence (P1 team vs P2 revealed)
    p1_species_set = set(p1_team_map.keys())
    p2_species_set = set(p2_hp.keys())
    for sp_name in TOP_META_SPECIES:
        in_p1 = 1.0 if sp_name in p1_species_set else 0.0
        in_p2 = 1.0 if sp_name in p2_species_set else 0.0
        feats[f"p1_has_{sp_name}"] = in_p1
        feats[f"p2_rev_{sp_name}"] = in_p2

    # Unrevealed Snorlax probability indicator (82.2% of teams carry Snorlax)
    feats["p2_likely_unrev_snorlax"] = 1.0 if ("snorlax" not in p2_species_set and n_unrev_p2 >= 1) else 0.0
    feats["p1_unseen_snorlax_alive"] = 1.0 if ("snorlax" in p1_species_set and "snorlax" not in p1_seen) else 0.0

    return feats


def extract_state_features(
    battle: Dict[str, Any],
    recon: Dict[str, Any],
    catalogue: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    """Family C: Current battle-state at Turn 30 & reconstructed 6v6 health/status/KO state."""
    feats: Dict[str, float] = {}
    tl = battle.get("battle_timeline", [])
    last_e = tl[-1] if tl else {}
    last_s1 = last_e.get("p1_pokemon_state") or {}
    last_s2 = last_e.get("p2_pokemon_state") or {}

    p1_team_map = recon["p1_team_map"]
    p1_hp = recon["p1_hp"]
    p1_status = recon["p1_status"]
    p1_moves_by_mon = recon["p1_moves_by_mon"]

    p2_hp = recon["p2_hp"]
    p2_status = recon["p2_status"]
    p2_moves_by_mon = recon["p2_moves_by_mon"]
    p2_exp_faints = recon["p2_explicit_faints"]
    p2_inf_kos = recon["p2_inferred_kos_strict"]
    p2_ko_probs = recon["p2_ko_probs"]
    last_seen_p1 = recon["last_seen_p1"]
    last_seen_p2 = recon["last_seen_p2"]

    n_unrev_p2 = max(0, 6 - len(p2_hp))

    # 1. P1 6-mon health & status aggregates
    p1_hps = np.array(list(p1_hp.values()), dtype=float)
    p1_alive_hps = p1_hps[p1_hps > 0.0]
    p1_total_hp = float(np.sum(p1_hps))
    p1_alive_cnt = float(len(p1_alive_hps))
    p1_fnt_cnt = float(np.sum(p1_hps <= 0.0))

    feats["state_p1_total_hp"] = p1_total_hp
    feats["state_p1_mean_hp"] = float(np.mean(p1_hps))
    feats["state_p1_mean_alive_hp"] = float(np.mean(p1_alive_hps)) if len(p1_alive_hps) > 0 else 0.0
    feats["state_p1_min_alive_hp"] = float(np.min(p1_alive_hps)) if len(p1_alive_hps) > 0 else 0.0
    feats["state_p1_max_alive_hp"] = float(np.max(p1_alive_hps)) if len(p1_alive_hps) > 0 else 0.0
    feats["state_p1_std_hp"] = float(np.std(p1_hps))
    feats["state_p1_alive_cnt"] = p1_alive_cnt
    feats["state_p1_fnt_cnt"] = p1_fnt_cnt
    feats["state_p1_full_hp_cnt"] = float(np.sum(p1_hps >= 0.99))
    feats["state_p1_healthy_cnt"] = float(np.sum(p1_hps >= 0.70))
    feats["state_p1_mid_hp_cnt"] = float(np.sum((p1_hps >= 0.30) & (p1_hps < 0.70)))
    feats["state_p1_crit_hp_cnt"] = float(np.sum((p1_hps > 0.0) & (p1_hps < 0.30)))

    # 2. P2 6-mon health & status aggregates (both raw and KO-adjusted)
    p2_raw_hps = list(p2_hp.values()) + [1.0] * n_unrev_p2
    p2_adj_hps = [
        0.0 if k in p2_inf_kos else v for k, v in p2_hp.items()
    ] + [1.0] * n_unrev_p2
    p2_soft_hps = [
        v * (1.0 - p2_ko_probs.get(k, 0.0)) for k, v in p2_hp.items()
    ] + [1.0] * n_unrev_p2

    p2_raw_arr = np.array(p2_raw_hps, dtype=float)
    p2_adj_arr = np.array(p2_adj_hps, dtype=float)
    p2_soft_arr = np.array(p2_soft_hps, dtype=float)
    p2_alive_adj_hps = p2_adj_arr[p2_adj_arr > 0.0]

    p2_total_hp_raw = float(np.sum(p2_raw_arr))
    p2_total_hp_adj = float(np.sum(p2_adj_arr))
    p2_total_hp_soft = float(np.sum(p2_soft_arr))
    p2_seen_hp_sum = float(sum(p2_hp.values()))

    p2_fnt_exp_cnt = float(len(p2_exp_faints))
    p2_fnt_adj_cnt = float(len(p2_inf_kos))
    p2_fnt_prob_sum = float(sum(p2_ko_probs.values()))
    p2_alive_exp_cnt = float(6.0 - p2_fnt_exp_cnt)
    p2_alive_adj_cnt = float(6.0 - p2_fnt_adj_cnt)
    p2_alive_soft_cnt = float(6.0 - p2_fnt_prob_sum)

    feats["state_p2_seen_hp_sum"] = p2_seen_hp_sum
    feats["state_p2_total_hp_raw"] = p2_total_hp_raw
    feats["state_p2_total_hp_adj"] = p2_total_hp_adj
    feats["state_p2_total_hp_soft"] = p2_total_hp_soft
    feats["state_p2_mean_hp_raw"] = float(np.mean(p2_raw_arr))
    feats["state_p2_mean_hp_adj"] = float(np.mean(p2_adj_arr))
    feats["state_p2_mean_alive_hp_adj"] = float(np.mean(p2_alive_adj_hps)) if len(p2_alive_adj_hps) > 0 else 0.0
    feats["state_p2_min_alive_hp_adj"] = float(np.min(p2_alive_adj_hps)) if len(p2_alive_adj_hps) > 0 else 0.0
    feats["state_p2_std_hp_raw"] = float(np.std(p2_raw_arr))
    feats["state_p2_fnt_exp_cnt"] = p2_fnt_exp_cnt
    feats["state_p2_fnt_inferred_cnt"] = float(len(p2_inf_kos - p2_exp_faints))
    feats["state_p2_fnt_adj_cnt"] = p2_fnt_adj_cnt
    feats["state_p2_fnt_prob_sum"] = p2_fnt_prob_sum
    feats["state_p2_alive_exp_cnt"] = p2_alive_exp_cnt
    feats["state_p2_alive_adj_cnt"] = p2_alive_adj_cnt
    feats["state_p2_alive_soft_cnt"] = p2_alive_soft_cnt
    feats["state_p2_full_hp_cnt"] = float(np.sum(p2_adj_arr >= 0.99))
    feats["state_p2_healthy_cnt"] = float(np.sum(p2_adj_arr >= 0.70))
    feats["state_p2_crit_hp_cnt"] = float(np.sum((p2_adj_arr > 0.0) & (p2_adj_arr < 0.30)))

    # Benched low-HP Pokémon that never returned before Turn 30
    p1_benched_low = sum(
        1.0 for m, h in p1_hp.items()
        if 0.0 < h <= 0.45 and last_seen_p1.get(m, 29) < 28
    )
    p2_benched_low = sum(
        1.0 for m, h in p2_hp.items()
        if 0.0 < h <= 0.45 and last_seen_p2.get(m, 29) < 28
    )
    feats["state_p1_benched_low_hp_cnt"] = float(p1_benched_low)
    feats["state_p2_benched_low_hp_cnt"] = float(p2_benched_low)
    feats["state_benched_low_hp_diff"] = float(p2_benched_low - p1_benched_low)

    # 3. Stat-weighted surviving HP for P1 and P2
    p1_wt_bst = 0.0
    p1_wt_off = 0.0
    p1_wt_bulk = 0.0
    p1_wt_spe = 0.0
    for m, h in p1_hp.items():
        if h > 0.0:
            sp = p1_team_map.get(m) or get_species_stats(m, catalogue)
            bst = sp["base_hp"] + sp["base_atk"] + sp["base_def"] + sp["base_spa"] + sp["base_spd"] + sp["base_spe"]
            off = max(sp["base_atk"], sp["base_spa"])
            blk = sp["base_hp"] + sp["base_def"] + sp["base_spd"]
            spe = sp["base_spe"] * (0.25 if p1_status.get(m) == "par" else 1.0)
            p1_wt_bst += h * bst
            p1_wt_off += h * off
            p1_wt_bulk += h * blk
            p1_wt_spe += h * spe

    p2_wt_bst_raw = n_unrev_p2 * 515.0
    p2_wt_bst_adj = n_unrev_p2 * 515.0
    p2_wt_off_adj = n_unrev_p2 * 105.0
    p2_wt_bulk_adj = n_unrev_p2 * 285.0
    p2_wt_spe_adj = n_unrev_p2 * 80.0
    for m, h in p2_hp.items():
        sp = get_species_stats(m, catalogue)
        bst = sp["base_hp"] + sp["base_atk"] + sp["base_def"] + sp["base_spa"] + sp["base_spd"] + sp["base_spe"]
        off = max(sp["base_atk"], sp["base_spa"])
        blk = sp["base_hp"] + sp["base_def"] + sp["base_spd"]
        spe = sp["base_spe"] * (0.25 if p2_status.get(m) == "par" else 1.0)
        p2_wt_bst_raw += h * bst
        h_adj = 0.0 if m in p2_inf_kos else h
        p2_wt_bst_adj += h_adj * bst
        p2_wt_off_adj += h_adj * off
        p2_wt_bulk_adj += h_adj * blk
        p2_wt_spe_adj += h_adj * spe

    feats["state_p1_wt_bst"] = p1_wt_bst
    feats["state_p2_wt_bst_raw"] = p2_wt_bst_raw
    feats["state_p2_wt_bst_adj"] = p2_wt_bst_adj
    feats["state_wt_bst_diff_raw"] = p1_wt_bst - p2_wt_bst_raw
    feats["state_wt_bst_diff_adj"] = p1_wt_bst - p2_wt_bst_adj
    feats["state_wt_off_diff_adj"] = p1_wt_off - p2_wt_off_adj
    feats["state_wt_bulk_diff_adj"] = p1_wt_bulk - p2_wt_bulk_adj
    feats["state_wt_spe_diff_adj"] = p1_wt_spe - p2_wt_spe_adj

    # Surviving Snorlax / top meta status at Turn 30
    feats["state_p1_snorlax_hp"] = float(p1_hp.get("snorlax", 0.0))
    p2_snorlax_hp = float(0.0 if "snorlax" in p2_inf_kos else p2_hp.get("snorlax", 0.82 if n_unrev_p2 >= 1 else 0.0))
    feats["state_p2_snorlax_hp_est"] = p2_snorlax_hp
    feats["state_snorlax_hp_diff"] = feats["state_p1_snorlax_hp"] - p2_snorlax_hp

    # 4. Status conditions across surviving Pokémon
    for side, st_map, hp_map, ko_set, mv_map in [
        ("p1", p1_status, p1_hp, set(), p1_moves_by_mon),
        ("p2", p2_status, p2_hp, p2_inf_kos, p2_moves_by_mon),
    ]:
        alive_mons = [m for m, h in hp_map.items() if h > 0.0 and m not in ko_set]
        for st_code in ("slp", "par", "tox", "psn", "brn", "frz"):
            feats[f"state_{side}_alive_{st_code}_cnt"] = float(
                sum(1 for m in alive_mons if st_map.get(m) == st_code)
            )
        feats[f"state_{side}_alive_any_status_cnt"] = float(
            sum(1 for m in alive_mons if st_map.get(m, "nostatus") not in ("nostatus", "fnt"))
        )
        # Crippled sleep (asleep without Rest or SleepTalk observed)
        crippled_slp = sum(
            1 for m in alive_mons
            if st_map.get(m) == "slp" and not ({"rest", "sleeptalk"} & mv_map.get(m, set()))
        )
        rest_slp = sum(
            1 for m in alive_mons
            if st_map.get(m) == "slp" and ({"rest", "sleeptalk"} & mv_map.get(m, set()))
        )
        feats[f"state_{side}_crippled_slp_cnt"] = float(crippled_slp)
        feats[f"state_{side}_rest_slp_cnt"] = float(rest_slp)

        # Clean (unstatused or RestTalk) remaining HP
        clean_hp = sum(
            hp_map[m] for m in alive_mons
            if st_map.get(m, "nostatus") == "nostatus"
            or (st_map.get(m) == "slp" and ({"rest", "sleeptalk"} & mv_map.get(m, set())))
        )
        if side == "p2":
            clean_hp += 1.0 * n_unrev_p2
        feats[f"state_{side}_clean_hp_sum"] = float(clean_hp)

    # 5. Active Pokémon state at Turn 30
    act_hp1 = 0.0 if last_s1.get("status") == "fnt" else float(last_s1.get("hp_pct", 1.0))
    act_hp2 = 0.0 if last_s2.get("status") == "fnt" else float(last_s2.get("hp_pct", 1.0))
    feats["state_p1_active_hp"] = act_hp1
    feats["state_p2_active_hp"] = act_hp2
    feats["state_active_hp_diff"] = act_hp1 - act_hp2
    feats["state_p1_active_fnt"] = 1.0 if last_s1.get("status") == "fnt" else 0.0
    feats["state_p2_active_fnt"] = 1.0 if last_s2.get("status") == "fnt" else 0.0

    b1_dict = last_s1.get("boosts") or {}
    b2_dict = last_s2.get("boosts") or {}
    for b_stat in ("atk", "def", "spa", "spd", "spe"):
        v1 = float(b1_dict.get(b_stat, 0))
        v2 = float(b2_dict.get(b_stat, 0))
        feats[f"state_p1_boost_{b_stat}"] = v1
        feats[f"state_p2_boost_{b_stat}"] = v2
        feats[f"state_boost_diff_{b_stat}"] = v1 - v2

    b1_sum = sum(float(v) for v in b1_dict.values())
    b2_sum = sum(float(v) for v in b2_dict.values())
    feats["state_p1_boost_sum"] = b1_sum
    feats["state_p2_boost_sum"] = b2_sum
    feats["state_boost_sum_diff"] = b1_sum - b2_sum

    eff1 = set(last_s1.get("effects") or [])
    eff2 = set(last_s2.get("effects") or [])
    feats["state_p1_has_sub"] = 1.0 if "substitute" in eff1 else 0.0
    feats["state_p2_has_sub"] = 1.0 if "substitute" in eff2 else 0.0
    feats["state_p1_perish"] = 1.0 if "perish" in eff1 else 0.0
    feats["state_p2_perish"] = 1.0 if "perish" in eff2 else 0.0

    # Field conditions (active Spikes)
    feats["state_p1_spikes_active"] = 1.0 if recon["p1_spikes_used"] > 0 else 0.0
    feats["state_p2_spikes_active"] = 1.0 if recon["p2_spikes_used"] > 0 else 0.0
    feats["state_spikes_diff"] = feats["state_p1_spikes_active"] - feats["state_p2_spikes_active"]

    # 6. Core Relative Advantage Features at Turn 30
    feats["state_hp_diff_raw"] = p1_total_hp - p2_total_hp_raw
    feats["state_hp_diff_adj"] = p1_total_hp - p2_total_hp_adj
    feats["state_hp_diff_soft"] = p1_total_hp - p2_total_hp_soft
    feats["state_hp_ratio_raw"] = (p1_total_hp + 0.25) / (p2_total_hp_raw + 0.25)
    feats["state_hp_ratio_adj"] = (p1_total_hp + 0.25) / (p2_total_hp_adj + 0.25)
    feats["state_hp_ratio_soft"] = (p1_total_hp + 0.25) / (p2_total_hp_soft + 0.25)
    feats["state_alive_diff_exp"] = p1_alive_cnt - p2_alive_exp_cnt
    feats["state_alive_diff_adj"] = p1_alive_cnt - p2_alive_adj_cnt
    feats["state_alive_diff_soft"] = p1_alive_cnt - p2_alive_soft_cnt
    feats["state_fnt_diff_exp"] = p2_fnt_exp_cnt - p1_fnt_cnt
    feats["state_fnt_diff_adj"] = p2_fnt_adj_cnt - p1_fnt_cnt
    feats["state_fnt_diff_soft"] = p2_fnt_prob_sum - p1_fnt_cnt
    feats["state_clean_hp_diff"] = feats["state_p1_clean_hp_sum"] - feats["state_p2_clean_hp_sum"]
    feats["state_status_disadv"] = feats["state_p1_alive_any_status_cnt"] - feats["state_p2_alive_any_status_cnt"]
    feats["state_crippled_slp_diff"] = feats["state_p2_crippled_slp_cnt"] - feats["state_p1_crippled_slp_cnt"]
    feats["state_healthy_cnt_diff"] = feats["state_p1_healthy_cnt"] - feats["state_p2_healthy_cnt"]

    # 7. Recency-weighted HP, P2 absence / phazed-absence scores, and Turn-30 sweep threat
    p1_recency_hp = 0.0
    for m, h in p1_hp.items():
        if m not in last_seen_p1 and h > 0.0:
            p1_recency_hp += 1.0
        elif h > 0.0:
            turns_ago = 29 - last_seen_p1.get(m, 29)
            p1_recency_hp += h * (0.985 ** max(0, turns_ago))

    p2_recency_hp = 1.0 * n_unrev_p2
    p2_weighted_absence = 0.0
    p2_long_gone_cnt = 0.0
    p2_phazed_absence_score = 0.0
    for m, h in p2_hp.items():
        if h > 0.0 and m not in p2_inf_kos:
            i_last = last_seen_p2.get(m, 29)
            turns_ago = max(0, 29 - i_last)
            decay = (0.94 ** turns_ago) if h < 0.65 else (0.975 ** turns_ago)
            p2_recency_hp += h * decay
            p2_weighted_absence += (1.0 - h) * turns_ago
            if turns_ago >= 10 and h <= 0.75:
                p2_long_gone_cnt += 1.0
            if i_last < len(tl) - 1:
                phazes_after = sum(
                    1 for j in range(i_last + 1, len(tl))
                    if tl[j].get("p1_move_details") and tl[j]["p1_move_details"]["name"] in PHAZING_MOVES
                )
                if phazes_after > 0:
                    p2_phazed_absence_score += phazes_after * (1.2 - h)

    feats["state_recency_hp_p1"] = float(p1_recency_hp)
    feats["state_recency_hp_p2"] = float(p2_recency_hp)
    feats["state_recency_hp_diff"] = float(p1_recency_hp - p2_recency_hp)
    feats["state_p2_weighted_absence"] = float(p2_weighted_absence)
    feats["state_p2_long_gone_cnt"] = float(p2_long_gone_cnt)
    feats["state_p2_phazed_absence_score"] = float(p2_phazed_absence_score)

    p1_off_boost = max(0.0, float(b1_dict.get("atk", 0))) + max(0.0, float(b1_dict.get("spa", 0)))
    p2_off_boost = max(0.0, float(b2_dict.get("atk", 0))) + max(0.0, float(b2_dict.get("spa", 0)))
    p1_def_boost = max(0.0, float(b1_dict.get("def", 0))) + max(0.0, float(b1_dict.get("spd", 0)))
    p2_def_boost = max(0.0, float(b2_dict.get("def", 0))) + max(0.0, float(b2_dict.get("spd", 0)))

    t30_p1_sweep = act_hp1 * (1.0 + 0.5 * p1_off_boost + 0.25 * p1_def_boost)
    t30_p2_sweep = act_hp2 * (1.0 + 0.5 * p2_off_boost + 0.25 * p2_def_boost)
    feats["state_t30_p1_sweep_threat"] = float(t30_p1_sweep)
    feats["state_t30_p2_sweep_threat"] = float(t30_p2_sweep)
    feats["state_t30_sweep_threat_diff"] = float(t30_p1_sweep - t30_p2_sweep)

    # Top-3 meta titan surviving HP differentials (Snorlax, Zapdos, Cloyster)
    feats["state_sp_hp_diff_snorlax"] = float(p1_hp.get("snorlax", 0.0) - p2_hp.get("snorlax", 0.0))
    feats["state_sp_hp_diff_zapdos"] = float(p1_hp.get("zapdos", 0.0) - p2_hp.get("zapdos", 0.0))
    feats["state_sp_hp_diff_cloyster"] = float(p1_hp.get("cloyster", 0.0) - p2_hp.get("cloyster", 0.0))

    return feats


def _linear_slope(y_arr: np.ndarray) -> float:
    """Fast closed-form OLS slope of y_arr against [0, 1, ..., len(y_arr)-1]."""
    n = len(y_arr)
    if n <= 1:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = 0.5 * (n - 1)
    y_mean = float(np.mean(y_arr))
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= 0.0:
        return 0.0
    return float(np.sum((x - x_mean) * (y_arr - y_mean)) / denom)


def extract_temporal_features(
    battle: Dict[str, Any],
    recon: Dict[str, Any],
) -> Dict[str, float]:
    """Family D: Multi-window temporal battle progression, momentum, and dynamics across 30 turns."""
    feats: Dict[str, float] = {}

    h1 = recon["h1_traj"]
    h2_raw = recon["h2_raw_traj"]
    h2_adj = recon["h2_adj_traj"]
    f1 = recon["f1_traj"]
    f2_exp = recon["f2_exp_traj"]
    f2_adj = recon["f2_adj_traj"]
    act_diff = recon["act_hp_diff_traj"]
    boost_diff = recon["boost_diff_traj"]

    hp_adv_raw = h1 - h2_raw
    hp_adv_adj = h1 - h2_adj
    fnt_adv_exp = f2_exp - f1
    fnt_adv_adj = f2_adj - f1

    # Snapshot advantages at milestone turns: T5, T10, T15, T20, T25, T30
    for t_idx, label in [(4, "t5"), (9, "t10"), (14, "t15"), (19, "t20"), (24, "t25"), (29, "t30")]:
        if t_idx < len(hp_adv_raw):
            feats[f"temp_hp_adv_raw_{label}"] = float(hp_adv_raw[t_idx])
            feats[f"temp_hp_adv_adj_{label}"] = float(hp_adv_adj[t_idx])
            feats[f"temp_fnt_adv_adj_{label}"] = float(fnt_adv_adj[t_idx])
            feats[f"temp_h1_{label}"] = float(h1[t_idx])
            feats[f"temp_h2_raw_{label}"] = float(h2_raw[t_idx])
            feats[f"temp_h2_adj_{label}"] = float(h2_adj[t_idx])

    # Window slices: W1 (1-10), W2 (11-20), W3 (21-30), Last 5 (26-30), Last 3 (28-30)
    windows = [
        ("w1_1_10", slice(0, 10)),
        ("w2_11_20", slice(10, 20)),
        ("w3_21_30", slice(20, 30)),
        ("last5", slice(25, 30)),
        ("last3", slice(27, 30)),
        ("full30", slice(0, 30)),
    ]

    p1_sw = recon["p1_switched_turn"]
    p2_sw = recon["p2_switched_turn"]
    p1_atk = recon["p1_attacked_turn"]
    p2_atk = recon["p2_attacked_turn"]
    p1_bp = recon["p1_move_bp_turn"]
    p2_bp = recon["p2_move_bp_turn"]
    p1_dmg = recon["p1_dmg_taken_turn"]
    p2_dmg = recon["p2_dmg_taken_turn"]
    p1_heal = recon["p1_healed_turn"]
    p2_heal = recon["p2_healed_turn"]

    for w_name, sl in windows:
        adv_r_w = hp_adv_raw[sl]
        adv_a_w = hp_adv_adj[sl]
        if len(adv_r_w) == 0:
            continue
        feats[f"temp_{w_name}_hp_adv_raw_mean"] = float(np.mean(adv_r_w))
        feats[f"temp_{w_name}_hp_adv_adj_mean"] = float(np.mean(adv_a_w))
        feats[f"temp_{w_name}_hp_adv_raw_slope"] = _linear_slope(adv_r_w)
        feats[f"temp_{w_name}_hp_adv_adj_slope"] = _linear_slope(adv_a_w)
        feats[f"temp_{w_name}_act_diff_mean"] = float(np.mean(act_diff[sl]))
        feats[f"temp_{w_name}_boost_diff_mean"] = float(np.mean(boost_diff[sl]))

        feats[f"temp_{w_name}_p1_switches"] = float(np.sum(p1_sw[sl]))
        feats[f"temp_{w_name}_p2_switches"] = float(np.sum(p2_sw[sl]))
        feats[f"temp_{w_name}_switch_diff"] = float(np.sum(p1_sw[sl]) - np.sum(p2_sw[sl]))

        feats[f"temp_{w_name}_p1_attacks"] = float(np.sum(p1_atk[sl]))
        feats[f"temp_{w_name}_p2_attacks"] = float(np.sum(p2_atk[sl]))
        feats[f"temp_{w_name}_attack_diff"] = float(np.sum(p1_atk[sl]) - np.sum(p2_atk[sl]))

        feats[f"temp_{w_name}_p1_bp_sum"] = float(np.sum(p1_bp[sl]))
        feats[f"temp_{w_name}_p2_bp_sum"] = float(np.sum(p2_bp[sl]))
        feats[f"temp_{w_name}_bp_diff"] = float(np.sum(p1_bp[sl]) - np.sum(p2_bp[sl]))

        feats[f"temp_{w_name}_p1_dmg_taken"] = float(np.sum(p1_dmg[sl]))
        feats[f"temp_{w_name}_p2_dmg_taken"] = float(np.sum(p2_dmg[sl]))
        feats[f"temp_{w_name}_dmg_dealt_adv"] = float(np.sum(p2_dmg[sl]) - np.sum(p1_dmg[sl]))

        feats[f"temp_{w_name}_p1_healed"] = float(np.sum(p1_heal[sl]))
        feats[f"temp_{w_name}_p2_healed"] = float(np.sum(p2_heal[sl]))
        feats[f"temp_{w_name}_heal_diff"] = float(np.sum(p1_heal[sl]) - np.sum(p2_heal[sl]))

    # Momentum & trend shifts between early, mid, and late battle
    if len(hp_adv_raw) >= 30:
        feats["temp_momentum_30_minus_20_raw"] = float(hp_adv_raw[29] - hp_adv_raw[19])
        feats["temp_momentum_30_minus_20_adj"] = float(hp_adv_adj[29] - hp_adv_adj[19])
        feats["temp_momentum_20_minus_10_raw"] = float(hp_adv_raw[19] - hp_adv_raw[9])
        feats["temp_momentum_20_minus_10_adj"] = float(hp_adv_adj[19] - hp_adv_adj[9])
        feats["temp_momentum_30_minus_10_raw"] = float(hp_adv_raw[29] - hp_adv_raw[9])
        feats["temp_momentum_30_minus_10_adj"] = float(hp_adv_adj[29] - hp_adv_adj[9])
        feats["temp_momentum_30_minus_25_raw"] = float(hp_adv_raw[29] - hp_adv_raw[24])
        feats["temp_momentum_30_minus_25_adj"] = float(hp_adv_adj[29] - hp_adv_adj[24])
        feats["temp_w3_minus_w1_adv_raw"] = feats["temp_w3_21_30_hp_adv_raw_mean"] - feats["temp_w1_1_10_hp_adv_raw_mean"]
        feats["temp_w3_minus_w1_adv_adj"] = feats["temp_w3_21_30_hp_adv_adj_mean"] - feats["temp_w1_1_10_hp_adv_adj_mean"]

        # Acceleration (late momentum minus early momentum)
        feats["temp_accel_raw"] = feats["temp_momentum_30_minus_20_raw"] - feats["temp_momentum_20_minus_10_raw"]
        feats["temp_accel_adj"] = feats["temp_momentum_30_minus_20_adj"] - feats["temp_momentum_20_minus_10_adj"]

        # Peak advantage, worst deficit, drawdown, and lead-change / reversal indicators
        feats["temp_max_hp_adv_raw"] = float(np.max(hp_adv_raw))
        feats["temp_min_hp_adv_raw"] = float(np.min(hp_adv_raw))
        feats["temp_drawdown_from_peak_raw"] = float(hp_adv_raw[29] - np.max(hp_adv_raw))
        feats["temp_comeback_from_min_raw"] = float(hp_adv_raw[29] - np.min(hp_adv_raw))
        feats["temp_turns_p1_ahead_raw"] = float(np.mean(hp_adv_raw > 0.0))
        feats["temp_turns_p1_ahead_adj"] = float(np.mean(hp_adv_adj > 0.0))
        feats["temp_turns_p1_ahead_last10"] = float(np.mean(hp_adv_raw[20:30] > 0.0))

        # Battle pace (total HP lost by both sides combined by turn 30)
        feats["temp_battle_pace_hp_lost"] = float((6.0 - h1[29]) + (6.0 - h2_adj[29]))
        feats["temp_total_faints_both"] = float(f1[29] + f2_adj[29])

        # Timing of first faint
        p1_first_fnt_turn = next((idx + 1 for idx, v in enumerate(f1) if v > 0), 35)
        p2_first_fnt_turn = next((idx + 1 for idx, v in enumerate(f2_adj) if v > 0), 35)
        feats["temp_p1_first_fnt_turn"] = float(p1_first_fnt_turn)
        feats["temp_p2_first_fnt_turn"] = float(p2_first_fnt_turn)
        feats["temp_first_fnt_turn_diff"] = float(p1_first_fnt_turn - p2_first_fnt_turn)

    return feats


def extract_matchup_features(
    battle: Dict[str, Any],
    recon: Dict[str, Any],
    catalogue: Dict[str, Dict[str, Any]],
    type_chart: Dict[Tuple[str, str], float],
) -> Dict[str, float]:
    """Family E: Generation 2 type chart effectiveness & stat matchups across active and surviving teams."""
    feats: Dict[str, float] = {}
    tl = battle.get("battle_timeline", [])
    last_e = tl[-1] if tl else {}
    last_s1 = last_e.get("p1_pokemon_state") or {}
    last_s2 = last_e.get("p2_pokemon_state") or {}

    p1_team_map = recon["p1_team_map"]
    p1_hp = recon["p1_hp"]
    p1_status = recon["p1_status"]
    p1_move_objs = recon["p1_move_objs_by_mon"]

    p2_hp = recon["p2_hp"]
    p2_status = recon["p2_status"]
    p2_inf_kos = recon["p2_inferred_kos_strict"]
    p2_move_objs = recon["p2_move_objs_by_mon"]

    # 1. Active Turn-30 matchup
    act_n1 = last_s1.get("name", "")
    act_n2 = last_s2.get("name", "")
    sp1_act = p1_team_map.get(act_n1) or get_species_stats(act_n1, catalogue)
    sp2_act = get_species_stats(act_n2, catalogue)

    t1_act = sp1_act["types"]
    t2_act = sp2_act["types"]

    # Best STAB multiplier of active P1 vs active P2 and vice versa
    p1_act_stab_vs_p2 = max(
        [get_type_effectiveness(t, t2_act, type_chart) for t in t1_act],
        default=1.0,
    )
    p2_act_stab_vs_p1 = max(
        [get_type_effectiveness(t, t1_act, type_chart) for t in t2_act],
        default=1.0,
    )
    feats["match_act_p1_stab_eff"] = float(p1_act_stab_vs_p2)
    feats["match_act_p2_stab_eff"] = float(p2_act_stab_vs_p1)
    feats["match_act_stab_eff_diff"] = float(p1_act_stab_vs_p2 - p2_act_stab_vs_p1)

    # Best observed damaging move multiplier of active P1 vs active P2
    m1_act_objs = p1_move_objs.get(act_n1, {}).values()
    m2_act_objs = p2_move_objs.get(act_n2, {}).values()
    p1_act_move_eff = max(
        [
            get_type_effectiveness(m.get("type"), t2_act, type_chart)
            * (1.5 if m.get("type") in t1_act else 1.0)
            for m in m1_act_objs
            if float(m.get("base_power", 0)) > 0
        ],
        default=p1_act_stab_vs_p2,
    )
    p2_act_move_eff = max(
        [
            get_type_effectiveness(m.get("type"), t1_act, type_chart)
            * (1.5 if m.get("type") in t2_act else 1.0)
            for m in m2_act_objs
            if float(m.get("base_power", 0)) > 0
        ],
        default=p2_act_stab_vs_p1,
    )
    feats["match_act_p1_move_eff"] = float(p1_act_move_eff)
    feats["match_act_p2_move_eff"] = float(p2_act_move_eff)
    feats["match_act_move_eff_diff"] = float(p1_act_move_eff - p2_act_move_eff)

    # Effective speed at Turn 30 (accounting for boosts and paralysis)
    b1_spe = int((last_s1.get("boosts") or {}).get("spe", 0))
    b2_spe = int((last_s2.get("boosts") or {}).get("spe", 0))
    eff_spe1 = sp1_act["base_spe"] * get_stage_multiplier(b1_spe) * (0.25 if last_s1.get("status") == "par" else 1.0)
    eff_spe2 = sp2_act["base_spe"] * get_stage_multiplier(b2_spe) * (0.25 if last_s2.get("status") == "par" else 1.0)
    feats["match_act_eff_spe_diff"] = float(eff_spe1 - eff_spe2)
    feats["match_act_p1_outspeeds"] = 1.0 if eff_spe1 > eff_spe2 else (0.5 if eff_spe1 == eff_spe2 else 0.0)

    # 2. Full Surviving Team Matchup Matrix at Turn 30
    p1_alive = [(m, h) for m, h in p1_hp.items() if h > 0.0]
    p2_alive = [(m, h) for m, h in p2_hp.items() if h > 0.0 and m not in p2_inf_kos]

    if p1_alive and p2_alive:
        p1_vs_p2_best_effs: List[float] = []
        p2_vs_p1_best_effs: List[float] = []
        p1_hp_wt_adv = 0.0
        outspeed_pairs = 0.0
        total_pairs = 0.0
        p1_se_count = 0.0
        p2_se_count = 0.0

        for m1_name, h1_val in p1_alive:
            sp1 = p1_team_map.get(m1_name) or get_species_stats(m1_name, catalogue)
            t1_list = sp1["types"]
            m1_list = [
                m for m in p1_move_objs.get(m1_name, {}).values()
                if float(m.get("base_power", 0)) > 0
            ]
            spe1 = sp1["base_spe"] * (0.25 if p1_status.get(m1_name) == "par" else 1.0)

            best_against_any_p2 = 0.0
            for m2_name, h2_val in p2_alive:
                sp2 = get_species_stats(m2_name, catalogue)
                t2_list = sp2["types"]
                m2_list = [
                    m for m in p2_move_objs.get(m2_name, {}).values()
                    if float(m.get("base_power", 0)) > 0
                ]
                spe2 = sp2["base_spe"] * (0.25 if p2_status.get(m2_name) == "par" else 1.0)

                # P1 m1 effectiveness against P2 m2
                eff12_stab = max([get_type_effectiveness(t, t2_list, type_chart) for t in t1_list], default=1.0)
                eff12_mv = max(
                    [get_type_effectiveness(m.get("type"), t2_list, type_chart) for m in m1_list],
                    default=eff12_stab,
                )
                eff12 = max(eff12_stab, eff12_mv)

                # P2 m2 effectiveness against P1 m1
                eff21_stab = max([get_type_effectiveness(t, t1_list, type_chart) for t in t2_list], default=1.0)
                eff21_mv = max(
                    [get_type_effectiveness(m.get("type"), t1_list, type_chart) for m in m2_list],
                    default=eff21_stab,
                )
                eff21 = max(eff21_stab, eff21_mv)

                p1_vs_p2_best_effs.append(eff12)
                p2_vs_p1_best_effs.append(eff21)
                best_against_any_p2 = max(best_against_any_p2, eff12)

                if eff12 >= 2.0:
                    p1_se_count += 1.0
                if eff21 >= 2.0:
                    p2_se_count += 1.0

                p1_hp_wt_adv += (h1_val * eff12) - (h2_val * eff21)
                if spe1 > spe2:
                    outspeed_pairs += 1.0
                elif spe1 == spe2:
                    outspeed_pairs += 0.5
                total_pairs += 1.0

        feats["match_surv_p1_mean_eff"] = float(np.mean(p1_vs_p2_best_effs))
        feats["match_surv_p2_mean_eff"] = float(np.mean(p2_vs_p1_best_effs))
        feats["match_surv_eff_diff"] = feats["match_surv_p1_mean_eff"] - feats["match_surv_p2_mean_eff"]
        feats["match_surv_hp_wt_type_adv"] = float(p1_hp_wt_adv / max(1.0, total_pairs))
        feats["match_surv_outspeed_frac"] = float(outspeed_pairs / max(1.0, total_pairs))
        feats["match_surv_se_frac_diff"] = float((p1_se_count - p2_se_count) / max(1.0, total_pairs))
    else:
        feats["match_surv_p1_mean_eff"] = 1.0
        feats["match_surv_p2_mean_eff"] = 1.0
        feats["match_surv_eff_diff"] = 0.0
        feats["match_surv_hp_wt_type_adv"] = 0.0
        feats["match_surv_outspeed_frac"] = 0.5
        feats["match_surv_se_frac_diff"] = 0.0

    # Grounded surviving Pokémon count (vulnerable to Spikes: not Flying-type)
    p1_grounded_alive = sum(
        1.0 for m, h in p1_alive
        if "flying" not in (p1_team_map.get(m) or get_species_stats(m, catalogue))["types"]
    )
    p2_grounded_alive = sum(
        1.0 for m, h in p2_alive
        if "flying" not in get_species_stats(m, catalogue)["types"]
    )
    feats["match_p1_grounded_alive"] = float(p1_grounded_alive)
    feats["match_p2_grounded_alive"] = float(p2_grounded_alive)

    return feats


def extract_interaction_features(feats: Dict[str, float]) -> Dict[str, float]:
    """Family F: Meaningful domain-grounded interaction features (Section 7F)."""
    inter: Dict[str, float] = {}

    hp_diff_raw = feats.get("state_hp_diff_raw", 0.0)
    hp_diff_adj = feats.get("state_hp_diff_adj", 0.0)
    alive_diff_adj = feats.get("state_alive_diff_adj", 0.0)
    fnt_diff_adj = feats.get("state_fnt_diff_adj", 0.0)
    wt_bst_diff = feats.get("state_wt_bst_diff_adj", 0.0)

    # 1. HP advantage × remaining Pokémon advantage
    inter["inter_hp_adv_x_alive_adv"] = hp_diff_adj * alive_diff_adj
    inter["inter_hp_raw_x_fnt_diff"] = hp_diff_raw * fnt_diff_adj
    inter["inter_combined_material_score"] = hp_diff_raw + 0.55 * fnt_diff_adj

    # 2. Type advantage × offensive potential
    type_adv = feats.get("match_surv_eff_diff", 0.0)
    off_diff = feats.get("state_wt_off_diff_adj", 0.0)
    inter["inter_type_adv_x_off_potential"] = type_adv * off_diff

    # 3. Health disadvantage × recovery capability
    p1_rec = feats.get("p1_has_direct_heal", 0.0) + feats.get("p1_resttalk_mon_cnt", 0.0)
    p2_rec = feats.get("p2_has_direct_heal", 0.0) + feats.get("p2_resttalk_mon_cnt", 0.0)
    inter["inter_hp_adv_x_recovery_diff"] = hp_diff_raw * (p1_rec - p2_rec)
    inter["inter_p1_deficit_x_recovery"] = min(0.0, hp_diff_raw) * p1_rec

    # 4. Team speed × observed offensive pressure
    outspeed_frac = feats.get("match_surv_outspeed_frac", 0.5)
    atk_diff = feats.get("temp_full30_attack_diff", 0.0)
    inter["inter_speed_x_off_pressure"] = (outspeed_frac - 0.5) * atk_diff

    # 5. Current advantage × recent momentum
    mom_last10 = feats.get("temp_momentum_30_minus_20_raw", 0.0)
    mom_last5 = feats.get("temp_momentum_30_minus_25_raw", 0.0)
    inter["inter_curr_adv_x_mom10"] = hp_diff_raw * mom_last10
    inter["inter_curr_adv_x_mom5"] = hp_diff_raw * mom_last5
    inter["inter_projected_hp_adv_t40"] = hp_diff_raw + 0.5 * mom_last10

    # 6. Status effects × remaining HP
    clean_hp_diff = feats.get("state_clean_hp_diff", 0.0)
    status_disadv = feats.get("state_status_disadv", 0.0)
    inter["inter_status_x_remaining_hp"] = clean_hp_diff - 0.35 * status_disadv

    # 7. Spikes × opponent grounded surviving Pokémon & switches
    p1_spikes = feats.get("state_p1_spikes_active", 0.0)
    p2_spikes = feats.get("state_p2_spikes_active", 0.0)
    p1_grounded = feats.get("match_p1_grounded_alive", 0.0)
    p2_grounded = feats.get("match_p2_grounded_alive", 0.0)
    inter["inter_spikes_x_grounded_diff"] = (p1_spikes * p2_grounded) - (p2_spikes * p1_grounded)

    return inter
