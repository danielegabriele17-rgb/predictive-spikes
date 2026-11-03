"""Core battle state reconstruction, Gen 2 mechanics, and per-battle feature families.

This module implements domain-specific feature extraction for Pokémon Generation 2 OU battles
from the first 30 observed turns (`battle_timeline`), organized into six modular families:
- Family A (`extract_team_features`): Player 1 full 6-Pokémon roster & Player 2 lead stats.
- Family B (`extract_p2_observed_features`): Player 2 revealed species, moves, and unrevealed backline.
- Family C (`extract_state_features`): Turn-30 health, status, stat boosts, and reconstructed KOs.
- Family D (`extract_temporal_features`): Multi-window trajectories, OLS slopes, and momentum.
- Family E (`extract_matchup_features`): Generation 2 type-chart effectiveness and speed matchups.
- Family F (`extract_interaction_features`): Non-linear domain interactions (material x momentum).
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from src.data_loader import get_type_effectiveness

# Standard 6 base statistics present in `p1_team_details` and `p2_lead_details`
STAT_NAMES: Tuple[str, ...] = ("hp", "atk", "def", "spa", "spd", "spe")

# Top 25 most frequent species in the Generation 2 OU competition metagame
TOP_META_SPECIES: Tuple[str, ...] = (
    "snorlax", "cloyster", "zapdos", "raikou", "tyranitar", "exeggutor", "gengar",
    "skarmory", "steelix", "machamp", "forretress", "starmie", "nidoking", "golem",
    "marowak", "umbreon", "vaporeon", "miltank", "jynx", "suicune", "blissey",
    "misdreavus", "heracross", "jolteon", "rhydon",
)

# All 17 elemental types active in Generation 2 (Fairy did not exist in Gen 2)
GEN2_TYPES: Tuple[str, ...] = (
    "normal", "electric", "ground", "rock", "steel", "water", "psychic", "ghost",
    "flying", "fighting", "grass", "ice", "fire", "dark", "poison", "bug", "dragon",
)

# Generation 2 tactical move categories used for behavioral feature extraction
DIRECT_HEAL_MOVES: Set[str] = {"rest", "recover", "softboiled", "milkdrink", "moonlight", "synthesis", "morningsun"}
RECOVERY_MOVES: Set[str] = DIRECT_HEAL_MOVES | {"painsplit", "leechseed"}
PHAZING_MOVES: Set[str] = {"roar", "whirlwind"}
SETUP_MOVES: Set[str] = {"curse", "bellydrum", "swordsdance", "growth", "agility", "amnesia", "barrier", "meditate", "sharpen", "minimize", "doubleteam"}
SELF_KO_MOVES: Set[str] = {"explosion", "selfdestruct", "destinybond", "perishsong"}
SLEEP_INDUCING_MOVES: Set[str] = {"sleeppowder", "lovelykiss", "hypnosis", "spore", "sing"}
PARA_INDUCING_MOVES: Set[str] = {"thunderwave", "stunspore", "glare", "bodyslam", "zapcannon"}
POISON_INDUCING_MOVES: Set[str] = {"toxic", "poisonpowder", "sludgebomb"}
FIXED_DAMAGE_MOVES: Set[str] = {"seismictoss", "nightshade", "counter", "mirrorcoat", "superfang", "fissure", "guillotine", "horndrill"}


def get_stage_multiplier(stage: int) -> float:
    """Compute Generation 2 stat stage multiplier for boost stages in [-6, +6].

    Positive stages multiply the stat by (2 + s) / 2; negative stages by 2 / (2 - s).
    """
    s = float(np.clip(stage, -6, 6))
    return (2.0 + s) / 2.0 if s >= 0.0 else 2.0 / (2.0 - s)


def _linear_slope(y_arr: np.ndarray) -> float:
    """Compute ordinary least-squares (OLS) linear trend slope over evenly spaced turns.

    Uses centered time indices `x = t - mean(t)` so `slope = dot(x, y - mean(y)) / dot(x, x)`.
    """
    n = len(y_arr)
    if n <= 1:
        return 0.0
    x = np.arange(n, dtype=float) - 0.5 * (n - 1)
    denom = float(np.dot(x, x))
    return float(np.dot(x, y_arr - float(np.mean(y_arr))) / denom) if denom > 0.0 else 0.0


def build_species_catalogue(battles: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build an internal Pokédex lookup mapping species `name` -> base stats & elemental types.

    Constructed strictly from competition JSONL data (`p1_team_details` and `p2_lead_details`)
    so that mid-battle switches by Player 2 can be mapped to their exact base stats and types
    without any external database.
    """
    catalogue: Dict[str, Dict[str, Any]] = {}
    for b in battles:
        mons = list(b.get("p1_team_details", []))
        if b.get("p2_lead_details"):
            mons.append(b["p2_lead_details"])
        for p in mons:
            name = p.get("name")
            if name and name not in catalogue:
                catalogue[name] = {
                    "types": [t for t in p.get("types", []) if t and t != "notype"],
                    **{f"base_{st}": float(p.get(f"base_{st}", 80)) for st in STAT_NAMES},
                }
    return catalogue


def get_species_stats(name: str, catalogue: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Retrieve species stats from the internal catalogue with a neutral fallback (80.0)."""
    if name in catalogue:
        return catalogue[name]
    return {"types": [], **{f"base_{st}": 80.0 for st in STAT_NAMES}}


def extract_team_features(battle: Dict[str, Any]) -> Dict[str, float]:
    """Family A: Extract Player 1 full 6-Pokémon roster stats & Player 2 lead matchup features.

    Uses vectorized 2D NumPy array aggregations (`mean`, `sum`, `min`, `max`, `std`) across
    all 6 base stats and `collections.Counter` for held items and elemental type coverage.
    """
    feats: Dict[str, float] = {}
    p1_team = battle.get("p1_team_details", [])
    p2_lead = battle.get("p2_lead_details") or {}

    # Build a (N_team, 6) NumPy matrix of P1 base stats for vectorized aggregation
    if p1_team:
        stats_mat = np.array(
            [[float(p.get(f"base_{st}", 0.0)) for st in STAT_NAMES] for p in p1_team],
            dtype=float,
        )
    else:
        stats_mat = np.zeros((1, len(STAT_NAMES)), dtype=float)

    stat_arrays: Dict[str, np.ndarray] = {}
    for col_idx, st in enumerate(STAT_NAMES):
        arr = stats_mat[:, col_idx]
        stat_arrays[st] = arr
        feats[f"p1_mean_{st}"] = float(np.mean(arr))
        feats[f"p1_sum_{st}"] = float(np.sum(arr))
        feats[f"p1_min_{st}"] = float(np.min(arr))
        feats[f"p1_max_{st}"] = float(np.max(arr))
        feats[f"p1_std_{st}"] = float(np.std(arr))

    # Base Stat Total (BST) per Pokémon = row-wise sum across the 6 stats
    bst_arr = np.sum(stats_mat, axis=1)
    for fn_name, fn in (("mean", np.mean), ("sum", np.sum), ("min", np.min), ("max", np.max), ("std", np.std)):
        feats[f"p1_{fn_name}_bst"] = float(fn(bst_arr))

    # Offensive vs defensive balance and speed/bulk tier counts
    off_arr = stat_arrays["atk"] + stat_arrays["spa"]
    def_arr = stat_arrays["def"] + stat_arrays["spd"] + stat_arrays["hp"]
    best_off_arr = np.maximum(stat_arrays["atk"], stat_arrays["spa"])
    feats["p1_mean_best_off"] = float(np.mean(best_off_arr))
    feats["p1_max_best_off"] = float(np.max(best_off_arr))
    feats["p1_off_def_ratio"] = float(np.sum(off_arr) / max(1.0, np.sum(def_arr)))
    feats["p1_phys_spec_ratio"] = float(np.sum(stat_arrays["atk"]) / max(1.0, np.sum(stat_arrays["spa"])))

    feats["p1_cnt_spe_ge_100"] = float(np.sum(stat_arrays["spe"] >= 100))
    feats["p1_cnt_spe_ge_115"] = float(np.sum(stat_arrays["spe"] >= 115))
    feats["p1_cnt_spe_le_60"] = float(np.sum(stat_arrays["spe"] <= 60))
    feats["p1_cnt_hp_ge_100"] = float(np.sum(stat_arrays["hp"] >= 100))
    feats["p1_cnt_def_ge_110"] = float(np.sum(stat_arrays["def"] >= 110))
    feats["p1_cnt_spd_ge_100"] = float(np.sum(stat_arrays["spd"] >= 100))

    # Level distribution and held item counts via collections.Counter
    levels = [float(p.get("level", 100)) for p in p1_team]
    feats["p1_mean_level"] = float(np.mean(levels)) if levels else 100.0
    feats["p1_is_all_lv100"] = 1.0 if all(lv == 100 for lv in levels) else 0.0

    items = [str(p.get("item", "")).lower() for p in p1_team]
    item_counts = Counter(items)
    feats["p1_leftovers_cnt"] = float(item_counts["leftovers"])
    feats["p1_berry_cnt"] = float(sum(1 for it in items if "berry" in it or it == "berryjuice"))
    feats["p1_thickclub_cnt"] = float(item_counts["thickclub"])
    feats["p1_unique_items_cnt"] = float(len(item_counts))

    # Elemental type diversity across P1's 6 Pokémon via collections.Counter
    p1_types_all = [t.lower() for p in p1_team for t in p.get("types", []) if t and t != "notype"]
    type_counts = Counter(p1_types_all)
    feats["p1_unique_types_cnt"] = float(len(type_counts))
    feats["p1_dual_type_cnt"] = float(sum(1 for p in p1_team if sum(1 for t in p.get("types", []) if t and t != "notype") == 2))
    for key_t in GEN2_TYPES:
        feats[f"p1_type_cnt_{key_t}"] = float(type_counts[key_t])

    # Player 2 lead base stats and Turn-1 lead speed/BST comparison
    for st in STAT_NAMES:
        feats[f"p2_lead_{st}"] = float(p2_lead.get(f"base_{st}", 0.0))
    p2_lead_bst = sum(feats[f"p2_lead_{st}"] for st in STAT_NAMES)
    feats["p2_lead_bst"] = p2_lead_bst
    feats["p2_lead_best_off"] = max(feats["p2_lead_atk"], feats["p2_lead_spa"])
    feats["p2_lead_bulk"] = feats["p2_lead_hp"] + feats["p2_lead_def"] + feats["p2_lead_spd"]
    feats["p2_lead_level"] = float(p2_lead.get("level", 100))

    feats["p1_vs_p2_lead_bst_diff"] = feats["p1_mean_bst"] - p2_lead_bst
    feats["p1_vs_p2_lead_spe_diff"] = feats["p1_mean_spe"] - feats["p2_lead_spe"]
    feats["p1_cnt_outspeed_p2_lead"] = float(np.sum(stat_arrays["spe"] > feats["p2_lead_spe"]))

    tl = battle.get("battle_timeline", [])
    p1_lead_name = tl[0]["p1_pokemon_state"]["name"] if tl else (p1_team[0]["name"] if p1_team else "")
    p1_lead_dict = next((p for p in p1_team if p.get("name") == p1_lead_name), p1_team[0] if p1_team else {})
    p1_lead_spe = float(p1_lead_dict.get("base_spe", 80.0))
    p1_lead_bst = sum(float(p1_lead_dict.get(f"base_{st}", 80.0)) for st in STAT_NAMES)
    feats["p1_lead_spe"] = p1_lead_spe
    feats["p1_lead_bst"] = p1_lead_bst
    feats["lead_spe_diff"] = p1_lead_spe - feats["p2_lead_spe"]
    feats["lead_outspeeds_p2"] = 1.0 if p1_lead_spe > feats["p2_lead_spe"] else (0.5 if p1_lead_spe == feats["p2_lead_spe"] else 0.0)
    feats["lead_bst_diff"] = p1_lead_bst - p2_lead_bst
    return feats


def reconstruct_battle_history(
    battle: Dict[str, Any],
    catalogue: Dict[str, Dict[str, Any]],
    type_chart: Dict[Tuple[str, str], float],
) -> Dict[str, Any]:
    """Single-pass reconstruction of the 30-turn battle timeline and Gen 2 physics KOs.

    Resolves two key artifacts of the competition's battle log parser:
    1. Cached `move_details` on voluntary switches: When a Pokémon switches in after having moved
       on an earlier turn, the parser often repeats its last cached `move_details`. We filter out
       these cached entries (`p1_executed` / `p2_executed`) to avoid double-counting attacks.
    2. Hidden Player 2 single-KOs on Turns 1–29: In the end-of-turn state snapshot, when a P2
       Pokémon faints from a single KO, its replacement immediately appears at the end of that
       turn, so `status == "fnt"` is never directly logged for the fainted Pokémon. We infer
       these KOs (`p2_inferred_kos_strict` and `p2_ko_probs`) when a damaged P2 Pokémon vanishes
       permanently while its replacement arrives untouched on a turn where P1 executed an attack,
       Explosion/Self-Destruct occurred, or fatal residual damage (Toxic/Burn/Leech Seed) applied.
    """
    tl = battle.get("battle_timeline", [])
    p1_team = battle.get("p1_team_details", [])
    p2_lead = battle.get("p2_lead_details") or {}

    # Map P1's 6 known Pokémon to their base stats, types, item, and level
    p1_team_map = {
        p.get("name", ""): {
            "types": [t for t in p.get("types", []) if t and t != "notype"],
            **{f"base_{st}": float(p.get(f"base_{st}", 80)) for st in STAT_NAMES},
            "item": str(p.get("item", "")).lower(),
            "level": float(p.get("level", 100)),
        }
        for p in p1_team
    }

    # Precompute the last turn index (0..29) on which each Pokémon name is observed
    last_seen_p1 = {e["p1_pokemon_state"]["name"]: i for i, e in enumerate(tl)}
    last_seen_p2 = {e["p2_pokemon_state"]["name"]: i for i, e in enumerate(tl)}

    # Initialize per-Pokémon HP, status, and observed move sets for P1 and P2
    p1_hp = {p["name"]: 1.0 for p in p1_team}
    p1_status = {p["name"]: "nostatus" for p in p1_team}
    p1_seen: Set[str] = set()
    p1_moves_by_mon: Dict[str, Set[str]] = {p["name"]: set() for p in p1_team}
    p1_move_objs_by_mon: Dict[str, Dict[str, Dict[str, Any]]] = {p["name"]: {} for p in p1_team}

    p2_lead_name = p2_lead.get("name", "")
    p2_hp = {p2_lead_name: 1.0} if p2_lead_name else {}
    p2_status = {p2_lead_name: "nostatus"} if p2_lead_name else {}
    p2_moves_by_mon: Dict[str, Set[str]] = {p2_lead_name: set()} if p2_lead_name else {}
    p2_move_objs_by_mon: Dict[str, Dict[str, Dict[str, Any]]] = {p2_lead_name: {}} if p2_lead_name else {}

    p2_explicit_faints: Set[str] = set()
    p2_inferred_kos_strict: Set[str] = set()
    p2_ko_probs: Dict[str, float] = {}

    last_cached_move_p1: Dict[str, str] = {}
    last_cached_move_p2: Dict[str, str] = {}
    prev_hp_p1, prev_hp_p2, prev_st_p2 = dict(p1_hp), dict(p2_hp), dict(p2_status)

    # Preallocate NumPy trajectory arrays across the observed turns
    n_turns = len(tl)
    h1_traj, h2_raw_traj, h2_adj_traj = np.zeros(n_turns), np.zeros(n_turns), np.zeros(n_turns)
    f1_traj, f2_exp_traj, f2_adj_traj = np.zeros(n_turns), np.zeros(n_turns), np.zeros(n_turns)
    act_hp_diff_traj, boost_diff_traj = np.zeros(n_turns), np.zeros(n_turns)
    p1_switched_turn, p2_switched_turn = np.zeros(n_turns), np.zeros(n_turns)
    p1_attacked_turn, p2_attacked_turn = np.zeros(n_turns), np.zeros(n_turns)
    p1_move_bp_turn, p2_move_bp_turn = np.zeros(n_turns), np.zeros(n_turns)
    p1_dmg_taken_turn, p2_dmg_taken_turn = np.zeros(n_turns), np.zeros(n_turns)
    p1_healed_turn, p2_healed_turn = np.zeros(n_turns), np.zeros(n_turns)

    p1_spikes = p2_spikes = p1_spin = p2_spin = p1_phaze = p2_phaze = p1_boom = p2_boom = 0

    for i, e in enumerate(tl):
        s1, s2 = e["p1_pokemon_state"], e["p2_pokemon_state"]
        m1, m2 = e.get("p1_move_details"), e.get("p2_move_details")
        n1, n2 = s1["name"], s2["name"]

        p1_seen.add(n1)
        if n2 not in p2_hp:
            p2_hp[n2], p2_status[n2], p2_moves_by_mon[n2], p2_move_objs_by_mon[n2] = 1.0, "nostatus", set(), {}

        cur_hp1 = 0.0 if s1["status"] == "fnt" else float(s1["hp_pct"])
        cur_hp2 = 0.0 if s2["status"] == "fnt" else float(s2["hp_pct"])

        # Detect non-faint active Pokémon transitions between turn i-1 and turn i
        p1_vol_switch = i > 0 and tl[i - 1]["p1_pokemon_state"]["name"] != n1 and tl[i - 1]["p1_pokemon_state"]["status"] != "fnt"
        p2_vol_switch = i > 0 and tl[i - 1]["p2_pokemon_state"]["name"] != n2 and tl[i - 1]["p2_pokemon_state"]["status"] != "fnt"
        if p1_vol_switch:
            p1_switched_turn[i] = 1.0
        if p2_vol_switch:
            p2_switched_turn[i] = 1.0

        # Filter out cached `move_details` repeated by the parser on voluntary switch-in turns
        p1_executed = False
        if m1 is not None:
            m1_name = m1["name"]
            p1_moves_by_mon.setdefault(n1, set()).add(m1_name)
            p1_move_objs_by_mon.setdefault(n1, {})[m1_name] = m1
            p1_executed = (not p1_vol_switch) or (m1_name != last_cached_move_p1.get(n1))
            last_cached_move_p1[n1] = m1_name

        p2_executed = False
        if m2 is not None:
            m2_name = m2["name"]
            p2_moves_by_mon.setdefault(n2, set()).add(m2_name)
            p2_move_objs_by_mon.setdefault(n2, {})[m2_name] = m2
            p2_executed = (not p2_vol_switch) or (m2_name != last_cached_move_p2.get(n2))
            last_cached_move_p2[n2] = m2_name

        # Track executed attacks, entry hazards (Spikes), hazard removal (Rapid Spin), and phazing
        if p1_executed and m1 is not None:
            m1_name, bp1 = m1["name"], float(m1.get("base_power", 0))
            if bp1 > 0 or m1_name in FIXED_DAMAGE_MOVES:
                p1_attacked_turn[i], p1_move_bp_turn[i] = 1.0, (bp1 if bp1 > 0 else 75.0)
            if m1_name == "spikes":
                p1_spikes += 1
            elif m1_name == "rapidspin":
                p1_spin += 1
                p2_spikes = 0
            elif m1_name in PHAZING_MOVES:
                p1_phaze += 1
            elif m1_name in {"explosion", "selfdestruct"}:
                p1_boom += 1

        if p2_executed and m2 is not None:
            m2_name, bp2 = m2["name"], float(m2.get("base_power", 0))
            if bp2 > 0 or m2_name in FIXED_DAMAGE_MOVES:
                p2_attacked_turn[i], p2_move_bp_turn[i] = 1.0, (bp2 if bp2 > 0 else 75.0)
            if m2_name == "spikes":
                p2_spikes += 1
            elif m2_name == "rapidspin":
                p2_spin += 1
                p1_spikes = 0
            elif m2_name in PHAZING_MOVES:
                p2_phaze += 1
            elif m2_name in {"explosion", "selfdestruct"}:
                p2_boom += 1

        # Record net damage taken and HP recovered on turn i
        old_hp1, old_hp2 = prev_hp_p1.get(n1, 1.0), prev_hp_p2.get(n2, 1.0)
        p1_dmg_taken_turn[i] = max(0.0, old_hp1 - cur_hp1)
        p1_healed_turn[i] = max(0.0, cur_hp1 - old_hp1)
        p2_dmg_taken_turn[i] = max(0.0, old_hp2 - cur_hp2)
        p2_healed_turn[i] = max(0.0, cur_hp2 - old_hp2)

        # Explicit P2 faint (logged on simultaneous double-KOs or on Turn 30)
        if s2["status"] == "fnt" or cur_hp2 <= 0.0:
            p2_explicit_faints.add(n2)
            p2_inferred_kos_strict.add(n2)
            p2_ko_probs[n2] = 1.0

        # Physics-based inference for P2 single-KOs on Turns 1..29:
        # If `n2_prev` is never seen again after turn `i-1` and replacement `n2` enters untouched
        if i > 0:
            s2_prev = tl[i - 1]["p2_pokemon_state"]
            n2_prev = s2_prev["name"]
            if n2 != n2_prev and s2_prev["status"] != "fnt" and float(s2_prev["hp_pct"]) > 0.0 and last_seen_p2.get(n2_prev, -1) == i - 1:
                exp_y_hp = prev_hp_p2.get(n2, 1.0)
                y_untouched = (cur_hp2 >= exp_y_hp - 0.13) and (s2["status"] == prev_st_p2.get(n2, "nostatus"))
                m1_name = m1["name"] if m1 else ""
                m1_bp = float(m1.get("base_power", 0)) if m1 else 0.0
                m1_type = m1.get("type", "notype") if m1 else "notype"
                prev_x_hp = float(s2_prev["hp_pct"])

                eff_vs_x = get_type_effectiveness(m1_type, get_species_stats(n2_prev, catalogue)["types"], type_chart) if m1_bp > 0 else 1.0
                eff_vs_y = get_type_effectiveness(m1_type, get_species_stats(n2, catalogue)["types"], type_chart) if m1_bp > 0 else 1.0
                m1_is_attack = p1_executed and m1_name not in PHAZING_MOVES and (m1_bp > 0 or m1_name in FIXED_DAMAGE_MOVES)
                p1_hp_drop = prev_hp_p1.get(n1, 1.0) - cur_hp1
                residual_death = (
                    (s2_prev["status"] in {"tox", "psn", "brn"} or any(ef in s2_prev.get("effects", []) for ef in ("leechseed", "perish", "curse", "nightmare")))
                    and prev_x_hp <= 0.28
                )

                if y_untouched:
                    if m1_is_attack and eff_vs_x > 0:
                        if prev_x_hp <= 0.55 or (prev_x_hp <= 0.80 and eff_vs_x >= 2.0):
                            p2_inferred_kos_strict.add(n2_prev)
                            p2_ko_probs[n2_prev] = max(p2_ko_probs.get(n2_prev, 0.0), 0.90)
                        elif eff_vs_y >= 1.0 and cur_hp2 >= exp_y_hp - 0.01 and float(m1.get("accuracy", 1.0)) >= 0.95:
                            p2_inferred_kos_strict.add(n2_prev)
                            p2_ko_probs[n2_prev] = max(p2_ko_probs.get(n2_prev, 0.0), 0.85)
                        else:
                            p2_ko_probs[n2_prev] = max(p2_ko_probs.get(n2_prev, 0.0), 0.55)
                    elif p1_hp_drop >= 0.22 and cur_hp2 >= 0.87:
                        p2_inferred_kos_strict.add(n2_prev)
                        p2_ko_probs[n2_prev] = max(p2_ko_probs.get(n2_prev, 0.0), 0.88)
                    elif residual_death:
                        p2_inferred_kos_strict.add(n2_prev)
                        p2_ko_probs[n2_prev] = max(p2_ko_probs.get(n2_prev, 0.0), 0.85)
                    elif prev_x_hp <= 0.20:
                        p2_ko_probs[n2_prev] = max(p2_ko_probs.get(n2_prev, 0.0), 0.65)
                elif prev_x_hp <= 0.15:
                    p2_ko_probs[n2_prev] = max(p2_ko_probs.get(n2_prev, 0.0), 0.35)

        # Update current state dictionaries and store turn-i trajectory values
        p1_hp[n1], p1_status[n1] = cur_hp1, s1["status"]
        p2_hp[n2], p2_status[n2] = cur_hp2, s2["status"]
        prev_hp_p1[n1], prev_hp_p2[n2], prev_st_p2[n2] = cur_hp1, cur_hp2, s2["status"]

        n_unrev_p2 = max(0, 6 - len(p2_hp))
        h1_traj[i] = sum(p1_hp.values())
        h2_raw_traj[i] = sum(p2_hp.values()) + 1.0 * n_unrev_p2
        h2_adj_traj[i] = sum(0.0 if k in p2_inferred_kos_strict else v for k, v in p2_hp.items()) + 1.0 * n_unrev_p2
        f1_traj[i] = sum(1.0 for v in p1_hp.values() if v <= 0.0)
        f2_exp_traj[i] = float(len(p2_explicit_faints))
        f2_adj_traj[i] = float(len(p2_inferred_kos_strict))
        act_hp_diff_traj[i] = cur_hp1 - cur_hp2
        boost_diff_traj[i] = float(sum(int(v) for v in (s1.get("boosts") or {}).values()) - sum(int(v) for v in (s2.get("boosts") or {}).values()))

    return {
        "p1_team_map": p1_team_map, "p1_hp": p1_hp, "p1_status": p1_status, "p1_seen": p1_seen,
        "p1_moves_by_mon": p1_moves_by_mon, "p1_move_objs_by_mon": p1_move_objs_by_mon,
        "p2_hp": p2_hp, "p2_status": p2_status, "p2_moves_by_mon": p2_moves_by_mon, "p2_move_objs_by_mon": p2_move_objs_by_mon,
        "p2_explicit_faints": p2_explicit_faints, "p2_inferred_kos_strict": p2_inferred_kos_strict, "p2_ko_probs": p2_ko_probs,
        "last_seen_p1": last_seen_p1, "last_seen_p2": last_seen_p2,
        "h1_traj": h1_traj, "h2_raw_traj": h2_raw_traj, "h2_adj_traj": h2_adj_traj,
        "f1_traj": f1_traj, "f2_exp_traj": f2_exp_traj, "f2_adj_traj": f2_adj_traj,
        "act_hp_diff_traj": act_hp_diff_traj, "boost_diff_traj": boost_diff_traj,
        "p1_switched_turn": p1_switched_turn, "p2_switched_turn": p2_switched_turn,
        "p1_attacked_turn": p1_attacked_turn, "p2_attacked_turn": p2_attacked_turn,
        "p1_move_bp_turn": p1_move_bp_turn, "p2_move_bp_turn": p2_move_bp_turn,
        "p1_dmg_taken_turn": p1_dmg_taken_turn, "p2_dmg_taken_turn": p2_dmg_taken_turn,
        "p1_healed_turn": p1_healed_turn, "p2_healed_turn": p2_healed_turn,
        "p1_spikes_used": p1_spikes, "p2_spikes_used": p2_spikes,
        "p1_rapidspin_used": p1_spin, "p2_rapidspin_used": p2_spin,
        "p1_phaze_used": p1_phaze, "p2_phaze_used": p2_phaze,
        "p1_explosion_used": p1_boom, "p2_explosion_used": p2_boom,
    }


def extract_p2_observed_features(
    battle: Dict[str, Any], recon: Dict[str, Any], catalogue: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    """Family B: Player 2 revealed roster, move categories, and unrevealed backline uncertainty.

    Infers base stats and types of all Pokémon revealed by Player 2 via `catalogue`,
    summarizes observed move categories (`physical`, `special`, `status`) using `Counter`,
    and encodes presence of top Gen 2 OU metagame staples (e.g., Snorlax, Zapdos, Cloyster).
    """
    p1_team_map, p2_hp, p1_seen = recon["p1_team_map"], recon["p2_hp"], recon["p1_seen"]
    p1_moves, p2_moves = recon["p1_moves_by_mon"], recon["p2_moves_by_mon"]
    p1_mobjs, p2_mobjs = recon["p1_move_objs_by_mon"], recon["p2_move_objs_by_mon"]

    n_rev_p2, n_seen_p1 = len(p2_hp), len(p1_seen)
    n_unrev_p2, n_unseen_p1 = max(0, 6 - n_rev_p2), max(0, 6 - n_seen_p1)
    feats: Dict[str, float] = {
        "p2_revealed_cnt": float(n_rev_p2), "p2_unrevealed_cnt": float(n_unrev_p2),
        "p1_seen_cnt": float(n_seen_p1), "p1_unseen_cnt": float(n_unseen_p1),
        "revealed_cnt_diff": float(n_rev_p2 - n_seen_p1),
    }

    # Build a 2D NumPy matrix of revealed P2 Pokémon stats from the species catalogue
    p2_sps = [get_species_stats(m, catalogue) for m in p2_hp]
    if p2_sps:
        rev_mat = np.array([[float(sp[f"base_{st}"]) for st in STAT_NAMES] for sp in p2_sps], dtype=float)
    else:
        rev_mat = np.full((1, len(STAT_NAMES)), 80.0, dtype=float)

    for col_idx, st in enumerate(STAT_NAMES):
        col = rev_mat[:, col_idx]
        feats[f"p2_rev_mean_{st}"] = float(np.mean(col))
        feats[f"p2_rev_max_{st}"] = float(np.max(col))
        feats[f"p2_rev_min_{st}"] = float(np.min(col))

    bst_arr = np.sum(rev_mat, axis=1)
    feats["p2_rev_mean_bst"] = float(np.mean(bst_arr))
    feats["p2_rev_max_bst"] = float(np.max(bst_arr))
    feats["p2_rev_min_bst"] = float(np.min(bst_arr))

    p2_rev_types = [t for sp in p2_sps for t in sp["types"]]
    feats["p2_rev_unique_types_cnt"] = float(len(set(p2_rev_types)))
    feats["p2_rev_cnt_spe_ge_100"] = float(np.sum(rev_mat[:, STAT_NAMES.index("spe")] >= 100.0)) if p2_sps else 0.0

    feats["p1_total_revealed_moves"] = float(sum(len(s) for s in p1_moves.values()))
    feats["p2_total_revealed_moves"] = float(sum(len(s) for s in p2_moves.values()))
    feats["p1_moves_per_seen_mon"] = feats["p1_total_revealed_moves"] / max(1.0, float(n_seen_p1))
    feats["p2_moves_per_rev_mon"] = feats["p2_total_revealed_moves"] / max(1.0, float(n_rev_p2))

    # Summarize observed move categories and tactical move sets for both sides
    for side, m_map, o_map in (("p1", p1_moves, p1_mobjs), ("p2", p2_moves, p2_mobjs)):
        mnames = set().union(*m_map.values()) if m_map else set()
        mobjs = [m for d in o_map.values() for m in d.values()]
        cat_counts = Counter(m.get("category") for m in mobjs)
        bps = [float(m.get("base_power", 0)) for m in mobjs if float(m.get("base_power", 0)) > 0]

        feats[f"{side}_obs_phys_moves_cnt"] = float(cat_counts["physical"])
        feats[f"{side}_obs_spec_moves_cnt"] = float(cat_counts["special"])
        feats[f"{side}_obs_stat_moves_cnt"] = float(cat_counts["status"])
        feats[f"{side}_obs_max_bp"] = float(max(bps)) if bps else 0.0
        feats[f"{side}_obs_mean_bp"] = float(np.mean(bps)) if bps else 0.0
        feats[f"{side}_has_recovery_move"] = float(len(mnames & RECOVERY_MOVES))
        feats[f"{side}_has_direct_heal"] = float(len(mnames & DIRECT_HEAL_MOVES))
        for mv in ("rest", "sleeptalk", "healbell", "spikes", "rapidspin", "curse"):
            feats[f"{side}_has_{mv}"] = 1.0 if mv in mnames else 0.0
        feats[f"{side}_has_phaze"] = float(len(mnames & PHAZING_MOVES))
        feats[f"{side}_has_setup"] = float(len(mnames & SETUP_MOVES))
        feats[f"{side}_has_sleep_move"] = float(len(mnames & SLEEP_INDUCING_MOVES))
        feats[f"{side}_has_para_move"] = float(len(mnames & PARA_INDUCING_MOVES))
        feats[f"{side}_has_poison_move"] = float(len(mnames & POISON_INDUCING_MOVES))
        feats[f"{side}_has_self_ko"] = float(len(mnames & SELF_KO_MOVES))

    feats["p1_resttalk_mon_cnt"] = float(sum(1 for s in p1_moves.values() if "rest" in s and "sleeptalk" in s))
    feats["p2_resttalk_mon_cnt"] = float(sum(1 for s in p2_moves.values() if "rest" in s and "sleeptalk" in s))

    # Binary indicators for top 25 Generation 2 OU metagame species
    p1_sp_set, p2_sp_set = set(p1_team_map.keys()), set(p2_hp.keys())
    for sp_name in TOP_META_SPECIES:
        feats[f"p1_has_{sp_name}"] = 1.0 if sp_name in p1_sp_set else 0.0
        feats[f"p2_rev_{sp_name}"] = 1.0 if sp_name in p2_sp_set else 0.0

    # Prior indicator: Snorlax is in >80% of Gen 2 OU teams, so if P2 has unrevealed Pokémon and hasn't shown Snorlax yet
    feats["p2_likely_unrev_snorlax"] = 1.0 if ("snorlax" not in p2_sp_set and n_unrev_p2 >= 1) else 0.0
    feats["p1_unseen_snorlax_alive"] = 1.0 if ("snorlax" in p1_sp_set and "snorlax" not in p1_seen) else 0.0
    return feats


def extract_state_features(
    battle: Dict[str, Any], recon: Dict[str, Any], catalogue: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    """Family C: Turn-30 material balance, reconstructed 6v6 HP/KOs, status severity, and boosts.

    Computes three complementary representations of Player 2's surviving material:
    - `raw`: Only explicit `"fnt"` statuses are zeroed out (unrevealed Pokémon count as 1.0 HP).
    - `adj`: Physics-inferred single KOs (`p2_inferred_kos_strict`) are also zeroed out.
    - `soft`: Each revealed P2 Pokémon's HP is weighted by its survival probability `1 - p2_ko_probs`.
    Also separates voluntary `Rest` sleep from enemy-induced crippled sleep (`Sleep Powder` / `Lovely Kiss`).
    """
    feats: Dict[str, float] = {}
    tl = battle.get("battle_timeline", [])
    last_e = tl[-1] if tl else {}
    last_s1, last_s2 = last_e.get("p1_pokemon_state") or {}, last_e.get("p2_pokemon_state") or {}

    p1_team_map, p1_hp, p1_status, p1_moves = recon["p1_team_map"], recon["p1_hp"], recon["p1_status"], recon["p1_moves_by_mon"]
    p2_hp, p2_status, p2_moves = recon["p2_hp"], recon["p2_status"], recon["p2_moves_by_mon"]
    p2_exp_faints, p2_inf_kos, p2_ko_probs = recon["p2_explicit_faints"], recon["p2_inferred_kos_strict"], recon["p2_ko_probs"]
    last_seen_p1, last_seen_p2 = recon["last_seen_p1"], recon["last_seen_p2"]
    n_unrev_p2 = max(0, 6 - len(p2_hp))

    # Player 1 surviving HP distribution across all 6 Pokémon
    p1_hps = np.array(list(p1_hp.values()), dtype=float)
    p1_alive_hps = p1_hps[p1_hps > 0.0]
    p1_total_hp, p1_alive_cnt, p1_fnt_cnt = float(np.sum(p1_hps)), float(len(p1_alive_hps)), float(np.sum(p1_hps <= 0.0))

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

    # Player 2 surviving HP distributions (raw, physics-adjusted, and soft probability-weighted)
    p2_raw_arr = np.array(list(p2_hp.values()) + [1.0] * n_unrev_p2, dtype=float)
    p2_adj_arr = np.array([0.0 if k in p2_inf_kos else v for k, v in p2_hp.items()] + [1.0] * n_unrev_p2, dtype=float)
    p2_soft_arr = np.array([v * (1.0 - p2_ko_probs.get(k, 0.0)) for k, v in p2_hp.items()] + [1.0] * n_unrev_p2, dtype=float)
    p2_alive_adj_hps = p2_adj_arr[p2_adj_arr > 0.0]

    p2_total_hp_raw, p2_total_hp_adj, p2_total_hp_soft = float(np.sum(p2_raw_arr)), float(np.sum(p2_adj_arr)), float(np.sum(p2_soft_arr))
    p2_fnt_exp_cnt, p2_fnt_adj_cnt, p2_fnt_prob_sum = float(len(p2_exp_faints)), float(len(p2_inf_kos)), float(sum(p2_ko_probs.values()))
    p2_alive_exp_cnt, p2_alive_adj_cnt, p2_alive_soft_cnt = 6.0 - p2_fnt_exp_cnt, 6.0 - p2_fnt_adj_cnt, 6.0 - p2_fnt_prob_sum

    feats["state_p2_seen_hp_sum"] = float(sum(p2_hp.values()))
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

    # Benched low-HP Pokémon that switched out before Turn 29
    p1_benched_low = sum(1.0 for m, h in p1_hp.items() if 0.0 < h <= 0.45 and last_seen_p1.get(m, 29) < 28)
    p2_benched_low = sum(1.0 for m, h in p2_hp.items() if 0.0 < h <= 0.45 and last_seen_p2.get(m, 29) < 28)
    feats["state_p1_benched_low_hp_cnt"] = float(p1_benched_low)
    feats["state_p2_benched_low_hp_cnt"] = float(p2_benched_low)
    feats["state_benched_low_hp_diff"] = float(p2_benched_low - p1_benched_low)

    # Quality-weighted remaining material (HP weighted by base stats and paralysis speed penalty)
    p1_wt_bst = p1_wt_off = p1_wt_bulk = p1_wt_spe = 0.0
    for m, h in p1_hp.items():
        if h > 0.0:
            sp = p1_team_map.get(m) or get_species_stats(m, catalogue)
            p1_wt_bst += h * sum(sp[f"base_{st}"] for st in STAT_NAMES)
            p1_wt_off += h * max(sp["base_atk"], sp["base_spa"])
            p1_wt_bulk += h * (sp["base_hp"] + sp["base_def"] + sp["base_spd"])
            p1_wt_spe += h * sp["base_spe"] * (0.25 if p1_status.get(m) == "par" else 1.0)

    # Unrevealed P2 Pokémon are assigned Gen 2 OU average base stats (BST=515, Off=105, Bulk=285, Spe=80)
    p2_wt_bst_raw = p2_wt_bst_adj = n_unrev_p2 * 515.0
    p2_wt_off_adj, p2_wt_bulk_adj, p2_wt_spe_adj = n_unrev_p2 * 105.0, n_unrev_p2 * 285.0, n_unrev_p2 * 80.0
    for m, h in p2_hp.items():
        sp = get_species_stats(m, catalogue)
        bst = sum(sp[f"base_{st}"] for st in STAT_NAMES)
        p2_wt_bst_raw += h * bst
        h_adj = 0.0 if m in p2_inf_kos else h
        p2_wt_bst_adj += h_adj * bst
        p2_wt_off_adj += h_adj * max(sp["base_atk"], sp["base_spa"])
        p2_wt_bulk_adj += h_adj * (sp["base_hp"] + sp["base_def"] + sp["base_spd"])
        p2_wt_spe_adj += h_adj * sp["base_spe"] * (0.25 if p2_status.get(m) == "par" else 1.0)

    feats["state_p1_wt_bst"] = p1_wt_bst
    feats["state_p2_wt_bst_raw"] = p2_wt_bst_raw
    feats["state_p2_wt_bst_adj"] = p2_wt_bst_adj
    feats["state_wt_bst_diff_raw"] = p1_wt_bst - p2_wt_bst_raw
    feats["state_wt_bst_diff_adj"] = p1_wt_bst - p2_wt_bst_adj
    feats["state_wt_off_diff_adj"] = p1_wt_off - p2_wt_off_adj
    feats["state_wt_bulk_diff_adj"] = p1_wt_bulk - p2_wt_bulk_adj
    feats["state_wt_spe_diff_adj"] = p1_wt_spe - p2_wt_spe_adj

    feats["state_p1_snorlax_hp"] = float(p1_hp.get("snorlax", 0.0))
    p2_snorlax_hp = float(0.0 if "snorlax" in p2_inf_kos else p2_hp.get("snorlax", 0.82 if n_unrev_p2 >= 1 else 0.0))
    feats["state_p2_snorlax_hp_est"] = p2_snorlax_hp
    feats["state_snorlax_hp_diff"] = feats["state_p1_snorlax_hp"] - p2_snorlax_hp

    # Status condition counts on surviving Pokémon via Counter, distinguishing Rest sleep from crippled sleep
    for side, st_map, hp_map, ko_set, mv_map in (("p1", p1_status, p1_hp, set(), p1_moves), ("p2", p2_status, p2_hp, p2_inf_kos, p2_moves)):
        alive_mons = [m for m, h in hp_map.items() if h > 0.0 and m not in ko_set]
        st_counts = Counter(st_map.get(m, "nostatus") for m in alive_mons)
        for st_code in ("slp", "par", "tox", "psn", "brn", "frz"):
            feats[f"state_{side}_alive_{st_code}_cnt"] = float(st_counts[st_code])
        feats[f"state_{side}_alive_any_status_cnt"] = float(sum(v for k, v in st_counts.items() if k not in ("nostatus", "fnt")))
        feats[f"state_{side}_crippled_slp_cnt"] = float(sum(1 for m in alive_mons if st_map.get(m) == "slp" and not ({"rest", "sleeptalk"} & mv_map.get(m, set()))))
        feats[f"state_{side}_rest_slp_cnt"] = float(sum(1 for m in alive_mons if st_map.get(m) == "slp" and ({"rest", "sleeptalk"} & mv_map.get(m, set()))))
        clean_hp = sum(hp_map[m] for m in alive_mons if st_map.get(m, "nostatus") == "nostatus" or (st_map.get(m) == "slp" and ({"rest", "sleeptalk"} & mv_map.get(m, set()))))
        feats[f"state_{side}_clean_hp_sum"] = float(clean_hp + (1.0 * n_unrev_p2 if side == "p2" else 0.0))

    # Active Pokémon HP, boosts, and volatile field effects at Turn 30
    act_hp1 = 0.0 if last_s1.get("status") == "fnt" else float(last_s1.get("hp_pct", 1.0))
    act_hp2 = 0.0 if last_s2.get("status") == "fnt" else float(last_s2.get("hp_pct", 1.0))
    feats["state_p1_active_hp"], feats["state_p2_active_hp"], feats["state_active_hp_diff"] = act_hp1, act_hp2, act_hp1 - act_hp2
    feats["state_p1_active_fnt"] = 1.0 if last_s1.get("status") == "fnt" else 0.0
    feats["state_p2_active_fnt"] = 1.0 if last_s2.get("status") == "fnt" else 0.0

    b1_dict, b2_dict = last_s1.get("boosts") or {}, last_s2.get("boosts") or {}
    for b_stat in ("atk", "def", "spa", "spd", "spe"):
        v1, v2 = float(b1_dict.get(b_stat, 0)), float(b2_dict.get(b_stat, 0))
        feats[f"state_p1_boost_{b_stat}"], feats[f"state_p2_boost_{b_stat}"], feats[f"state_boost_diff_{b_stat}"] = v1, v2, v1 - v2

    b1_sum, b2_sum = sum(float(v) for v in b1_dict.values()), sum(float(v) for v in b2_dict.values())
    feats["state_p1_boost_sum"], feats["state_p2_boost_sum"], feats["state_boost_sum_diff"] = b1_sum, b2_sum, b1_sum - b2_sum

    eff1, eff2 = set(last_s1.get("effects") or []), set(last_s2.get("effects") or [])
    feats["state_p1_has_sub"] = 1.0 if "substitute" in eff1 else 0.0
    feats["state_p2_has_sub"] = 1.0 if "substitute" in eff2 else 0.0
    feats["state_p1_perish"] = 1.0 if "perish" in eff1 else 0.0
    feats["state_p2_perish"] = 1.0 if "perish" in eff2 else 0.0

    feats["state_p1_spikes_active"] = 1.0 if recon["p1_spikes_used"] > 0 else 0.0
    feats["state_p2_spikes_active"] = 1.0 if recon["p2_spikes_used"] > 0 else 0.0
    feats["state_spikes_diff"] = feats["state_p1_spikes_active"] - feats["state_p2_spikes_active"]

    # Primary material differentials and regularized HP ratios (top-ranked features in XGBoost gain)
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

    # Recency-discounted HP and phazed-absence score (damaged P2 Pokémon that never got pulled in by Roar/Whirlwind)
    p1_recency_hp = sum(1.0 if m not in last_seen_p1 else h * (0.985 ** max(0, 29 - last_seen_p1.get(m, 29))) for m, h in p1_hp.items() if h > 0.0)
    p2_recency_hp = 1.0 * n_unrev_p2
    p2_weighted_absence = p2_long_gone_cnt = p2_phazed_absence_score = 0.0
    for m, h in p2_hp.items():
        if h > 0.0 and m not in p2_inf_kos:
            i_last = last_seen_p2.get(m, 29)
            turns_ago = max(0, 29 - i_last)
            p2_recency_hp += h * ((0.94 ** turns_ago) if h < 0.65 else (0.975 ** turns_ago))
            p2_weighted_absence += (1.0 - h) * turns_ago
            if turns_ago >= 10 and h <= 0.75:
                p2_long_gone_cnt += 1.0
            if i_last < len(tl) - 1:
                phazes_after = sum(1 for j in range(i_last + 1, len(tl)) if tl[j].get("p1_move_details") and tl[j]["p1_move_details"]["name"] in PHAZING_MOVES)
                if phazes_after > 0:
                    p2_phazed_absence_score += phazes_after * (1.2 - h)

    feats["state_recency_hp_p1"] = float(p1_recency_hp)
    feats["state_recency_hp_p2"] = float(p2_recency_hp)
    feats["state_recency_hp_diff"] = float(p1_recency_hp - p2_recency_hp)
    feats["state_p2_weighted_absence"] = float(p2_weighted_absence)
    feats["state_p2_long_gone_cnt"] = float(p2_long_gone_cnt)
    feats["state_p2_phazed_absence_score"] = float(p2_phazed_absence_score)

    # Turn-30 active setup sweeper threat index (HP * offensive/defensive boost multiplier)
    p1_off_boost = max(0.0, float(b1_dict.get("atk", 0))) + max(0.0, float(b1_dict.get("spa", 0)))
    p2_off_boost = max(0.0, float(b2_dict.get("atk", 0))) + max(0.0, float(b2_dict.get("spa", 0)))
    p1_def_boost = max(0.0, float(b1_dict.get("def", 0))) + max(0.0, float(b1_dict.get("spd", 0)))
    p2_def_boost = max(0.0, float(b2_dict.get("def", 0))) + max(0.0, float(b2_dict.get("spd", 0)))
    t30_p1_sweep = act_hp1 * (1.0 + 0.5 * p1_off_boost + 0.25 * p1_def_boost)
    t30_p2_sweep = act_hp2 * (1.0 + 0.5 * p2_off_boost + 0.25 * p2_def_boost)
    feats["state_t30_p1_sweep_threat"], feats["state_t30_p2_sweep_threat"], feats["state_t30_sweep_threat_diff"] = float(t30_p1_sweep), float(t30_p2_sweep), float(t30_p1_sweep - t30_p2_sweep)

    for sp_t in ("snorlax", "zapdos", "cloyster"):
        feats[f"state_sp_hp_diff_{sp_t}"] = float(p1_hp.get(sp_t, 0.0) - p2_hp.get(sp_t, 0.0))
    return feats


def extract_temporal_features(battle: Dict[str, Any], recon: Dict[str, Any]) -> Dict[str, float]:
    """Family D: Multi-window temporal battle progression, OLS slopes, and momentum across 30 turns.

    Partitions the 30-turn timeline into early (`Turns 1–10`), mid (`11–20`), late (`21–30`),
    and end-window (`Last 5`, `Last 3`, `Full 30`) slices to measure trajectory velocity and acceleration.
    """
    feats: Dict[str, float] = {}
    h1, h2_raw, h2_adj = recon["h1_traj"], recon["h2_raw_traj"], recon["h2_adj_traj"]
    f1, f2_adj = recon["f1_traj"], recon["f2_adj_traj"]
    act_diff, boost_diff = recon["act_hp_diff_traj"], recon["boost_diff_traj"]

    hp_adv_raw, hp_adv_adj, fnt_adv_adj = h1 - h2_raw, h1 - h2_adj, f2_adj - f1

    # Snapshot checkpoints every 5 turns (t5, t10, t15, t20, t25, t30)
    for t_idx, label in ((4, "t5"), (9, "t10"), (14, "t15"), (19, "t20"), (24, "t25"), (29, "t30")):
        if t_idx < len(hp_adv_raw):
            feats[f"temp_hp_adv_raw_{label}"] = float(hp_adv_raw[t_idx])
            feats[f"temp_hp_adv_adj_{label}"] = float(hp_adv_adj[t_idx])
            feats[f"temp_fnt_adv_adj_{label}"] = float(fnt_adv_adj[t_idx])
            feats[f"temp_h1_{label}"] = float(h1[t_idx])
            feats[f"temp_h2_raw_{label}"] = float(h2_raw[t_idx])
            feats[f"temp_h2_adj_{label}"] = float(h2_adj[t_idx])

    # Windowed means, OLS slopes, and action totals (switches, attacks, base power, damage, healing)
    windows = (("w1_1_10", slice(0, 10)), ("w2_11_20", slice(10, 20)), ("w3_21_30", slice(20, 30)), ("last5", slice(25, 30)), ("last3", slice(27, 30)), ("full30", slice(0, 30)))
    p1_sw, p2_sw = recon["p1_switched_turn"], recon["p2_switched_turn"]
    p1_atk, p2_atk = recon["p1_attacked_turn"], recon["p2_attacked_turn"]
    p1_bp, p2_bp = recon["p1_move_bp_turn"], recon["p2_move_bp_turn"]
    p1_dmg, p2_dmg = recon["p1_dmg_taken_turn"], recon["p2_dmg_taken_turn"]
    p1_heal, p2_heal = recon["p1_healed_turn"], recon["p2_healed_turn"]

    for w_name, sl in windows:
        adv_r_w, adv_a_w = hp_adv_raw[sl], hp_adv_adj[sl]
        if len(adv_r_w) == 0:
            continue
        feats[f"temp_{w_name}_hp_adv_raw_mean"] = float(np.mean(adv_r_w))
        feats[f"temp_{w_name}_hp_adv_adj_mean"] = float(np.mean(adv_a_w))
        feats[f"temp_{w_name}_hp_adv_raw_slope"] = _linear_slope(adv_r_w)
        feats[f"temp_{w_name}_hp_adv_adj_slope"] = _linear_slope(adv_a_w)
        feats[f"temp_{w_name}_act_diff_mean"] = float(np.mean(act_diff[sl]))
        feats[f"temp_{w_name}_boost_diff_mean"] = float(np.mean(boost_diff[sl]))
        for key, a1, a2, flip in (
            ("switches", p1_sw, p2_sw, False), ("attacks", p1_atk, p2_atk, False),
            ("bp_sum", p1_bp, p2_bp, False), ("dmg_taken", p1_dmg, p2_dmg, True), ("healed", p1_heal, p2_heal, False),
        ):
            s1_v, s2_v = float(np.sum(a1[sl])), float(np.sum(a2[sl]))
            feats[f"temp_{w_name}_p1_{key}"], feats[f"temp_{w_name}_p2_{key}"] = s1_v, s2_v
            diff_name = "dmg_dealt_adv" if key == "dmg_taken" else ("heal_diff" if key == "healed" else f"{key.replace('_sum', '').replace('es', '').rstrip('s')}_diff")
            feats[f"temp_{w_name}_{diff_name}"] = (s2_v - s1_v) if flip else (s1_v - s2_v)

    # First- and second-order momentum, peak drawdown, comeback, and first-KO timing via np.flatnonzero
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
        feats["temp_accel_raw"] = feats["temp_momentum_30_minus_20_raw"] - feats["temp_momentum_20_minus_10_raw"]
        feats["temp_accel_adj"] = feats["temp_momentum_30_minus_20_adj"] - feats["temp_momentum_20_minus_10_adj"]
        feats["temp_max_hp_adv_raw"] = float(np.max(hp_adv_raw))
        feats["temp_min_hp_adv_raw"] = float(np.min(hp_adv_raw))
        feats["temp_drawdown_from_peak_raw"] = float(hp_adv_raw[29] - np.max(hp_adv_raw))
        feats["temp_comeback_from_min_raw"] = float(hp_adv_raw[29] - np.min(hp_adv_raw))
        feats["temp_turns_p1_ahead_raw"] = float(np.mean(hp_adv_raw > 0.0))
        feats["temp_turns_p1_ahead_adj"] = float(np.mean(hp_adv_adj > 0.0))
        feats["temp_turns_p1_ahead_last10"] = float(np.mean(hp_adv_raw[20:30] > 0.0))
        feats["temp_battle_pace_hp_lost"] = float((6.0 - h1[29]) + (6.0 - h2_adj[29]))
        feats["temp_total_faints_both"] = float(f1[29] + f2_adj[29])

        nz1, nz2 = np.flatnonzero(f1 > 0), np.flatnonzero(f2_adj > 0)
        p1_ff = int(nz1[0]) + 1 if len(nz1) > 0 else 35
        p2_ff = int(nz2[0]) + 1 if len(nz2) > 0 else 35
        feats["temp_p1_first_fnt_turn"], feats["temp_p2_first_fnt_turn"], feats["temp_first_fnt_turn_diff"] = float(p1_ff), float(p2_ff), float(p1_ff - p2_ff)
    return feats


def extract_matchup_features(
    battle: Dict[str, Any], recon: Dict[str, Any], catalogue: Dict[str, Dict[str, Any]], type_chart: Dict[Tuple[str, str], float],
) -> Dict[str, float]:
    """Family E: Generation 2 type-chart effectiveness & speed matchups across active and surviving rosters.

    Computes STAB (1.5x Same-Type Attack Bonus), best revealed move multiplier, boost-adjusted and
    paralysis-adjusted (0.25x) active speed superiority, and all-pairs surviving roster type dominance.
    """
    tl = battle.get("battle_timeline", [])
    last_e = tl[-1] if tl else {}
    last_s1, last_s2 = last_e.get("p1_pokemon_state") or {}, last_e.get("p2_pokemon_state") or {}
    p1_team_map, p1_hp, p1_status, p1_move_objs = recon["p1_team_map"], recon["p1_hp"], recon["p1_status"], recon["p1_move_objs_by_mon"]
    p2_hp, p2_status, p2_inf_kos, p2_move_objs = recon["p2_hp"], recon["p2_status"], recon["p2_inferred_kos_strict"], recon["p2_move_objs_by_mon"]

    # Turn-30 active-vs-active STAB effectiveness, revealed move effectiveness, and effective speed
    act_n1, act_n2 = last_s1.get("name", ""), last_s2.get("name", "")
    sp1_act = p1_team_map.get(act_n1) or get_species_stats(act_n1, catalogue)
    sp2_act = get_species_stats(act_n2, catalogue)
    t1_act, t2_act = sp1_act["types"], sp2_act["types"]

    p1_act_stab = max([get_type_effectiveness(t, t2_act, type_chart) for t in t1_act], default=1.0)
    p2_act_stab = max([get_type_effectiveness(t, t1_act, type_chart) for t in t2_act], default=1.0)
    p1_act_mv = max([get_type_effectiveness(m.get("type"), t2_act, type_chart) * (1.5 if m.get("type") in t1_act else 1.0) for m in p1_move_objs.get(act_n1, {}).values() if float(m.get("base_power", 0)) > 0], default=p1_act_stab)
    p2_act_mv = max([get_type_effectiveness(m.get("type"), t1_act, type_chart) * (1.5 if m.get("type") in t2_act else 1.0) for m in p2_move_objs.get(act_n2, {}).values() if float(m.get("base_power", 0)) > 0], default=p2_act_stab)

    eff_spe1 = sp1_act["base_spe"] * get_stage_multiplier(int((last_s1.get("boosts") or {}).get("spe", 0))) * (0.25 if last_s1.get("status") == "par" else 1.0)
    eff_spe2 = sp2_act["base_spe"] * get_stage_multiplier(int((last_s2.get("boosts") or {}).get("spe", 0))) * (0.25 if last_s2.get("status") == "par" else 1.0)

    feats: Dict[str, float] = {
        "match_act_p1_stab_eff": float(p1_act_stab), "match_act_p2_stab_eff": float(p2_act_stab),
        "match_act_stab_eff_diff": float(p1_act_stab - p2_act_stab),
        "match_act_p1_move_eff": float(p1_act_mv), "match_act_p2_move_eff": float(p2_act_mv),
        "match_act_move_eff_diff": float(p1_act_mv - p2_act_mv),
        "match_act_eff_spe_diff": float(eff_spe1 - eff_spe2),
        "match_act_p1_outspeeds": 1.0 if eff_spe1 > eff_spe2 else (0.5 if eff_spe1 == eff_spe2 else 0.0),
    }

    # Pairwise type and speed matchup matrix across all surviving P1 and P2 Pokémon
    p1_alive = [(m, h) for m, h in p1_hp.items() if h > 0.0]
    p2_alive = [(m, h) for m, h in p2_hp.items() if h > 0.0 and m not in p2_inf_kos]
    if p1_alive and p2_alive:
        p1_effs, p2_effs = [], []
        p1_hp_wt_adv = outspeed_pairs = total_pairs = p1_se = p2_se = 0.0
        for m1_name, h1_val in p1_alive:
            sp1 = p1_team_map.get(m1_name) or get_species_stats(m1_name, catalogue)
            t1_list = sp1["types"]
            m1_list = [m for m in p1_move_objs.get(m1_name, {}).values() if float(m.get("base_power", 0)) > 0]
            spe1 = sp1["base_spe"] * (0.25 if p1_status.get(m1_name) == "par" else 1.0)
            for m2_name, h2_val in p2_alive:
                sp2 = get_species_stats(m2_name, catalogue)
                t2_list = sp2["types"]
                m2_list = [m for m in p2_move_objs.get(m2_name, {}).values() if float(m.get("base_power", 0)) > 0]
                spe2 = sp2["base_spe"] * (0.25 if p2_status.get(m2_name) == "par" else 1.0)
                eff12 = max(max([get_type_effectiveness(t, t2_list, type_chart) for t in t1_list], default=1.0), max([get_type_effectiveness(m.get("type"), t2_list, type_chart) for m in m1_list], default=0.0))
                eff21 = max(max([get_type_effectiveness(t, t1_list, type_chart) for t in t2_list], default=1.0), max([get_type_effectiveness(m.get("type"), t1_list, type_chart) for m in m2_list], default=0.0))
                p1_effs.append(eff12)
                p2_effs.append(eff21)
                p1_se += 1.0 if eff12 >= 2.0 else 0.0
                p2_se += 1.0 if eff21 >= 2.0 else 0.0
                p1_hp_wt_adv += (h1_val * eff12) - (h2_val * eff21)
                outspeed_pairs += 1.0 if spe1 > spe2 else (0.5 if spe1 == spe2 else 0.0)
                total_pairs += 1.0
        feats["match_surv_p1_mean_eff"] = float(np.mean(p1_effs))
        feats["match_surv_p2_mean_eff"] = float(np.mean(p2_effs))
        feats["match_surv_eff_diff"] = feats["match_surv_p1_mean_eff"] - feats["match_surv_p2_mean_eff"]
        feats["match_surv_hp_wt_type_adv"] = float(p1_hp_wt_adv / max(1.0, total_pairs))
        feats["match_surv_outspeed_frac"] = float(outspeed_pairs / max(1.0, total_pairs))
        feats["match_surv_se_frac_diff"] = float((p1_se - p2_se) / max(1.0, total_pairs))
    else:
        feats.update({"match_surv_p1_mean_eff": 1.0, "match_surv_p2_mean_eff": 1.0, "match_surv_eff_diff": 0.0, "match_surv_hp_wt_type_adv": 0.0, "match_surv_outspeed_frac": 0.5, "match_surv_se_frac_diff": 0.0})

    # Count grounded surviving Pokémon (non-Flying in Gen 2, as Levitate was introduced in Gen 3) vulnerable to Spikes
    feats["match_p1_grounded_alive"] = float(sum(1.0 for m, _ in p1_alive if "flying" not in (p1_team_map.get(m) or get_species_stats(m, catalogue))["types"]))
    feats["match_p2_grounded_alive"] = float(sum(1.0 for m, _ in p2_alive if "flying" not in get_species_stats(m, catalogue)["types"]))
    return feats


def extract_interaction_features(feats: Dict[str, float]) -> Dict[str, float]:
    """Family F: Non-linear domain interactions combining material, momentum, and Spikes pressure.

    Includes `inter_combined_material_score` (the #1 highest-gain feature across all XGBoost models),
    projected Turn-40 HP advantage, and Spikes x grounded-targets differential.
    """
    hp_raw, hp_adj = feats.get("state_hp_diff_raw", 0.0), feats.get("state_hp_diff_adj", 0.0)
    alive_adj, fnt_adj = feats.get("state_alive_diff_adj", 0.0), feats.get("state_fnt_diff_adj", 0.0)
    p1_rec = feats.get("p1_has_direct_heal", 0.0) + feats.get("p1_resttalk_mon_cnt", 0.0)
    p2_rec = feats.get("p2_has_direct_heal", 0.0) + feats.get("p2_resttalk_mon_cnt", 0.0)
    mom10, mom5 = feats.get("temp_momentum_30_minus_20_raw", 0.0), feats.get("temp_momentum_30_minus_25_raw", 0.0)
    return {
        "inter_hp_adv_x_alive_adv": hp_adj * alive_adj,
        "inter_hp_raw_x_fnt_diff": hp_raw * fnt_adj,
        "inter_combined_material_score": hp_raw + 0.55 * fnt_adj,
        "inter_type_adv_x_off_potential": feats.get("match_surv_eff_diff", 0.0) * feats.get("state_wt_off_diff_adj", 0.0),
        "inter_hp_adv_x_recovery_diff": hp_raw * (p1_rec - p2_rec),
        "inter_p1_deficit_x_recovery": min(0.0, hp_raw) * p1_rec,
        "inter_speed_x_off_pressure": (feats.get("match_surv_outspeed_frac", 0.5) - 0.5) * feats.get("temp_full30_attack_diff", 0.0),
        "inter_curr_adv_x_mom10": hp_raw * mom10,
        "inter_curr_adv_x_mom5": hp_raw * mom5,
        "inter_projected_hp_adv_t40": hp_raw + 0.5 * mom10,
        "inter_status_x_remaining_hp": feats.get("state_clean_hp_diff", 0.0) - 0.35 * feats.get("state_status_disadv", 0.0),
        "inter_spikes_x_grounded_diff": (feats.get("state_p1_spikes_active", 0.0) * feats.get("match_p2_grounded_alive", 0.0)) - (feats.get("state_p2_spikes_active", 0.0) * feats.get("match_p1_grounded_alive", 0.0)),
    }
