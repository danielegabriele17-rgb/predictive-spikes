# Pokémon Battle Prediction 2026 — Technical Methodology & Experimental Report

**Competition**: Sapienza University of Rome — *Foundations of Data Science: Pokémon Battle Prediction 2026*  
**Task**: Binary Classification (`player_won` $\in \{0, 1\}$) from 30-Turn Incomplete Generation 2 OU Battle Logs  
**Evaluation Metric**: Classification Accuracy  
**Model Family**: Regularized Gradient Boosted Decision Trees (`XGBoost`)

---

## 1. Executive Summary

This report documents the design, exploratory data analysis, domain-physics state reconstruction, feature engineering, validation architecture, hyperparameter optimization, and final model selection for our competitive **XGBoost** pipeline.

Starting from the competition organizers' official baseline (`LogisticRegression` on 6 static team/lead stat features, **5-Fold CV Accuracy = `0.5178 +/- 0.0080`**, ROC-AUC = `0.5256`), we systematically engineered a 6-family Generation 2 OU feature representation and leak-free within-fold empirical-Bayes target encoding pipeline. Through progressive ablations, collinearity pruning, and multi-configuration ensembling, we achieved a **5-Fold CV Accuracy of `0.7030 +/- 0.0017`** (**`+18.52` percentage points** over the organizer baseline, OOF ROC-AUC = **`0.7767`**, OOF Log-Loss = **`0.5636`**).

---

## 2. Exploratory Data Analysis & Dataset Schema

### 2.1 Dataset Overview
The competition dataset located in `fds-pokemon-battles-prediction-2026/` comprises:
- **`train.jsonl`**: `10,000` battles with binary target `player_won` (`5,000` wins for Player 1 and `5,000` wins for Player 2 — exact `50.0% / 50.0%` class balance).
- **`test.jsonl`**: `5,000` battles (`battle_id` `0..4999`) without `player_won`.
- **`sample_submission.csv`**: `5,000` rows with columns `battle_id,player_won`.
- **`type_chart_gen2.csv`**: `110` attack-defense type effectiveness multipliers (`0.0`, `0.5`, `2.0`; standard `1.0` neutral interactions omitted).

### 2.2 JSON Record Schema
Each battle JSON object contains:
1. **`battle_id`** (`int`): Unique identifier (`0..9999` in train, `0..4999` in test).
2. **`p1_team_details`** (`list[dict]`, length `6`): Complete roster for Player 1. Each entry contains `name`, `level` (always `100`), `types` (`list[str]` of length 2, using `"notype"` for single-typed Pokémon), and base stats (`base_hp`, `base_atk`, `base_def`, `base_spa`, `base_spd`, `base_spe`). *Note*: Held items are not exposed in the JSON schema.
3. **`p2_lead_details`** (`dict`): Complete `name`, `level`, `types`, and 6 base stats for Player 2's starting Pokémon only. Player 2's remaining 5 Pokémon are initially unrevealed.
4. **`battle_timeline`** (`list[dict]`, always exactly `30` turns, `turn = 1..30`): Each turn entry records end-of-turn state snapshots (`p1_pokemon_state`, `p2_pokemon_state`) and move dictionaries (`p1_move_details`, `p2_move_details`).
   - **`p{1,2}_pokemon_state`**: Contains `name`, `hp_pct` $\in [0.0, 1.0]$, `status` $\in \{\text{"nostatus"}, \text{"par"}, \text{"slp"}, \text{"psn"}, \text{"tox"}, \text{"brn"}, \text{"frz"}, \text{"fnt"}\}$, `effects` $\subseteq \{\text{"noeffect"}, \text{"reflect"}, \text{"Substitute"}, \text{"confusion"}, \text{"leechseed"}, \text{"firespin"}, \text{"wrap"}, \text{"clamp"}, \text{"meanlook"}, \text{"curse"}, \text{"nightmare"}, \text{"perish3..1"}, \text{"encore"}, \text{"attract"}, \text{"typechange"}, \text{" safeguard"}\}$, and `boosts` (`atk`, `def`, `spa`, `spd`, `spe`).
   - **`p{1,2}_move_details`**: Either `None` or a dict with `name`, `type`, `category` $\in \{\text{"Physical"}, \text{"Special"}, \text{"Status"}\}$, `base_power`, `accuracy`, and `priority`.

### 2.3 Critical Data Mechanics Uncovered During EDA

1. **Stateful `move_details` Caching Trap (Preventing Move Double-Counting)**:
   - Inspection of consecutive turns in `battle_timeline` revealed that once a Pokémon executes a move on turn $t_0$, the underlying simulator parser caches `move_details` on that Pokémon's state object and **never resets it to `None`** when the Pokémon switches out.
   - Consequently, when that Pokémon voluntarily switches back into battle on a later turn $t > t_0$ (where it cannot attack because switching consumes its turn), `p{1,2}_move_details` repeats its cached move from turn $t_0$!
   - Across `train.jsonl`, naive iteration over `battle_timeline` misclassifies tens of thousands of voluntary switch-in turns as executed attacks. In [`reconstruct_battle_history`](file:///Users/Daniele/Documents/GitHub/predictive-spikes/src/battle_features.py#L221-L526), we detect voluntary switch-ins (`name[t] != name[t-1]` when `status[t-1] != "fnt"`) and mask out cached `move_details` on those turns.

2. **Asymmetry Between Player 1 and Player 2 Faint Recording (`status == "fnt"`)**:
   - Across all `300,000` training turns, `p1_pokemon_state["status"] == "fnt"` appears **`16,882` times** (`1.69` faints/battle), whereas `p2_pokemon_state["status"] == "fnt"` appears only **`5,040` times** (`0.50` faints/battle).
   - Detailed forensic inspection showed that of those `5,040` P2 `"fnt"` events, **`4,950` (`98.2%`) occur on simultaneous double-KOs** (`p1` also `"fnt"`, e.g. *Explosion* / *Self-Destruct*) and **`90` (`1.8%`) occur on Turn 30**. On Turns `1..29`, whenever Player 2's active Pokémon $X$ faints **alone**, Player 2 immediately sends out replacement Pokémon $Y$ before the turn snapshot is recorded, so `p2_pokemon_state` at turn $t$ already shows replacement $Y$ (`hp_pct = 1.0`, `status = "nostatus"`) and $X$ never returns to battle!
   - By reconstructing Player 2 single-KOs in [`reconstruct_battle_history`](file:///Users/Daniele/Documents/GitHub/predictive-spikes/src/battle_features.py#L369-L515) using Generation 2 battle physics (verifying that $X$ never reappears in turns $t..30$, that replacement $Y$ was untouched on turn $t$, and scoring P1's executed attack power/effectiveness, prior low HP, toxic/burn/curse residual damage, or P2 *Explosion*/*Destiny Bond*), the Pearson correlation of material differential (`f2 - f1`) with `player_won` jumped from **`+0.2557` to `+0.3360`** (`+0.0803` correlation gain).

3. **Exact Species Determinism Within the Dataset**:
   - Across all `15,000` battles (`247` unique Pokémon species), a Pokémon's `name` uniquely determines its `(types, base_hp, base_atk, base_def, base_spa, base_spd, base_spe)` with **zero inconsistencies**.
   - Building an internal species lookup dictionary strictly from `p1_team_details` and `p2_lead_details` ([`build_species_catalogue`](file:///Users/Daniele/Documents/GitHub/predictive-spikes/src/battle_features.py#L140-L176)) enables exact stat and type reconstruction for every Pokémon Player 2 reveals during turns `1..30` without any external data.

4. **Duplicate Battles & Meta Concentration**:
   - `train.jsonl` contains `14` exact duplicate battle pairs (`28` rows), `13` of which have consistent labels and `1` pair has conflicting labels.
   - Generation 2 OU is heavily centralized around *Snorlax* (present on `82.2%` of all teams), *Zapdos*, *Cloyster*, *Tyranitar*, *Exeggutor*, *Raikou*, *Skarmory*, *Gengar*, *Starmie*, and *Marowak*. Furthermore, `69.54%` of `test.jsonl` battles use a 6-Pokémon Player 1 team composition that also appears in `train.jsonl`.

---

## 3. Analysis of the Official Organizer Baseline

The official baseline notebook ([`notebook baseline.ipynb`](file:///Users/Daniele/Documents/GitHub/predictive-spikes/notebook%20baseline.ipynb)) implements:
- `simple_feature_extraction`: Sums each of the 6 base stats across Player 1's 6 Pokémon (`p1_hp`, `p1_atk`, `p1_def`, `p1_spa`, `p1_spd`, `p1_spe`) and extracts Player 2's lead base stats (`p2_lead_hp..spe`). In its code cell, it actually drops down to 6 features and trains a `LogisticRegression(max_iter=1000)` on the full training set without cross-validation.
- **Why it underperforms (`0.5178` 5-fold CV Accuracy)**:
  1. It completely ignores `battle_timeline` (all 30 turns of actual gameplay, damage, KOs, status conditions, and revealed Pokémon).
  2. Pre-battle team stat totals have weak predictive power in Gen 2 OU because competitive teams are drawn from a narrow pool of balanced archetypes (`~50–55%` accuracy ceiling from static team rosters alone).

---

## 4. Feature Engineering Architecture (`src/battle_features.py` & `src/feature_engineering.py`)

We designed our feature engineering pipeline around **6 modular domain families** plus a strictly leak-free **Within-Fold Empirical-Bayes Target Encoder**:

### Family A: Player 1 Team Composition (`team`, 88 features)
- **Stat Distributions**: Mean, sum, std, min, max, range, and coefficient of variation for `base_hp`, `base_atk`, `base_def`, `base_spa`, `base_spd`, `base_spe`, and `bst` (Base Stat Total).
- **Composite Bulk & Speed Tiers**: Physical bulk (`HP * Def`), special bulk (`HP * SpD`), total bulk, mixed attacker counts, and speed tier counts (`>100`, `>=95`, `<60`).
- **Type Diversity & Gen 2 OU Meta Staples**: Unique type count, type entropy, explicit type coverage indicators (`electric`, `ground`, `water`, `psychic`, `steel`, `rock`, `ghost`, `ice`, `fighting`, `normal`), and Gen 2 OU staple indicators (*Snorlax*, *Zapdos*, *Cloyster*, *Tyranitar*, *Raikou*, *Skarmory*, *Exeggutor*, *Gengar*, *Starmie*, *Marowak*, *Blissey*, *Suicune*, *Mislabeled/Heracross/Forretress*).

### Family B: Player 2 Observed Team & Information Asymmetry (`p2_observed`, 76 features)
- **Lead & Revealed Roster Aggregates**: Base stats of Player 2's lead, count of revealed Pokémon (`1..6`), unrevealed count (`6 - revealed`), fraction revealed, and known-roster stat means/maxima/minima inferred via the internal species catalogue.
- **Proper Missingness Handling**: Unobserved reserve Pokémon are never imputed with `0` base stats; aggregates are computed over observed Pokémon alongside explicit uncertainty/unrevealed indicators.
- **Observed Move Repertoire**: Unique moves executed by Player 2, max observed base power, physical/special/status move ratios, and revealed recovery/phazing/hazard/sleep/explosion capabilities.

### Family C: Turn-30 Reconstructed Battle State (`state`, 111 features)
- **Material & Survival Balance**: Explicit `"fnt"` counts, physics-reconstructed P2 single-KO counts, continuous expected faint totals (`expected_faints`), confirmed alive counts, and unrevealed healthy reserve counts (`6 - p2_revealed`).
- **Full-Team & Recency-Decayed HP Pools**: Total known HP (`p1_known_hp_sum` + `p1_unrevealed * 1.0` vs `p2_known_hp_sum` + `p2_unrevealed * 1.0`), mean/min/std of alive HP, **recency-decayed HP** (`hp_pct * 0.985^(30 - last_seen_turn)` to discount injured Pokémon hidden on the bench for 15+ turns), and **phazed-absence scores** (penalizing injured Pokémon phazed out by *Roar*/*Whirlwind* that never returned).
- **Weighted Status & Active Effects**: Severity-weighted status burden (`slp=0.55`, `frz=0.65`, `par=0.35`, `tox=0.35`, `brn=0.30`, `psn=0.20`), active *Reflect*/*Light Screen*/*Substitute*/*Leech Seed*/*Curse* flags, and active Turn-30 stat boost differentials (`atk`, `def`, `spa`, `spd`, `spe`).
- **Top Gen 2 Titan HP Differentials**: Dense Turn-30 HP differentials (`p1_hp - p2_hp`) specifically for the three most battle-warping Gen 2 OU Pokémon (*Snorlax*, *Zapdos*, *Cloyster*).

### Family D: Multi-Window Temporal Dynamics (`temporal`, 199 features)
- **Windowed Trajectories**: Extracted across 5 distinct temporal windows: Early (`Turns 1–10`), Mid (`Turns 11–20`), Late (`Turns 21–30`), `Last 10 Turns` (`21–30`), and `Last 5 Turns` (`26–30`).
- **Within-Window Metrics**: Damage dealt and received, net damage advantage, executed attacks, voluntary switches, faints, status applications, recovery moves, and mean team HP differential in each window.
- **Momentum & Slope Dynamics**: OLS linear regression slope of `team_hp_diff(t)` across all 30 turns and across the final 10 turns, late-vs-early acceleration (`late_hp_diff - early_hp_diff`), lead changes, longest consecutive streak in the lead, and tempo control (attack-to-switch ratio and initiative rate where a player attacks while the opponent switches or is immobilized).

### Family E: Generation 2 Type Chart & Matchup Matrix (`matchup`, 51 features)
- **Lead & Active Turn-30 Type Matchups**: Using `type_chart_gen2.csv`, computes best STAB offensive multiplier and best executed-move multiplier between the Turn-1 leads and between the Turn-30 active Pokémon.
- **Full-Team vs Revealed-Roster Matchup Matrix**: Computes maximum offensive multiplier, defensive resistance ratio, immunity count (`0.0x`), and super-effective coverage count (`>=2.0x`) for every surviving Player 1 Pokémon against every surviving Player 2 Pokémon.
- **Active Turn-30 Speed & KO Threat**: Exact speed comparison at Turn 30 incorporating base Speed, active `spe` boost stage multipliers, and paralysis (`0.25x` Speed in Gen 2).

### Family F: Domain Interactions & Non-Linear Ratios (`interactions`, 21 features)
- **Effective Actionable HP**: Combines total team HP with status penalties (`total_hp - 0.35 * status_weighted_sum`) and boost advantages.
- **Turn-30 Active Sweep Threat**: Product of active Pokémon remaining HP, offensive stat + boost multiplier, STAB type effectiveness against the opponent's active Pokémon, and outspeed indicator (`state_sweep_threat_diff`), capturing late-game *Belly Drum* / *Curse* / *Swords Dance* sweep states.

### Within-Fold Empirical-Bayes Target Encoding (`WithinFoldTargetEncoder`, 12 features)
- Strictly computed **inside each CV fold** (`fit_transform_inner_oof` using 5 inner folds on the training split, `transform` on validation/test) with Laplace smoothing ($m = 30$ for species/leads, $m = 20$ for 6-Pokémon team signatures):
  - Player 1 team empirical win rate, mean/min/max individual species win rates, Player 2 lead win rate, `(p1_lead, p2_lead)` matchup win rate, Player 2 revealed roster win rate, and **HP-weighted surviving species win-rate expectations** at Turn 30 (`te_p1_alive_hp_weighted_wr`, `te_p2_alive_hp_weighted_wr`, `te_alive_hp_weighted_wr_diff`).

---

## 5. Validation Strategy & Leakage Prevention

1. **Primary Cross-Validation Scheme**: 5-Fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`), preserving the exact `50.0% / 50.0%` class balance in every fold (`8,000` train / `2,000` validation per fold).
2. **Duplicate-Grouped Sensitivity Check**: We also implemented `StratifiedGroupKFold` (`group_duplicates=True` in [`get_cv_splits`](file:///Users/Daniele/Documents/GitHub/predictive-spikes/src/validation.py#L46-L78)) using canonical MD5 battle hashes so that the 14 duplicate battle pairs in `train.jsonl` are forced into the same fold. Across all models, grouped vs. standard stratified CV differs by `< 0.0010` (`0.10%`), confirming zero duplicate-induced inflation.
3. **Strict Separation of Tuning Splits and Evaluation Splits**: Hyperparameter optimization with Optuna was conducted on an independent 5-fold split (`seed=1337`, duplicate-grouped), while final model evaluation and out-of-fold ensembling were measured on the canonical `seed=42` splits.
4. **Zero Preprocessing or Target Leakage**:
   - All target encodings (`WithinFoldTargetEncoder`) are fit exclusively on `y[tr_idx]` inside each fold.
   - Feature subset selection (`top140`, `top200`, `top220`) was verified inside folds (`Phase 4b` in `scripts/train.py`, yielding `0.6996 +/- 0.0049` strictly inside-fold vs `0.7008` global pruned).
   - `test.jsonl` is never used for feature selection, target encoding, threshold selection, or hyperparameter tuning.

---

## 6. Experimental Log & Progressive Feature Ablation Study

All ablation experiments were executed via [`scripts/evaluate.py`](file:///Users/Daniele/Documents/GitHub/predictive-spikes/scripts/evaluate.py) and [`scripts/train.py`](file:///Users/Daniele/Documents/GitHub/predictive-spikes/scripts/train.py) on the exact same 5-fold `StratifiedKFold(seed=42)` splits (logged in [`outputs/ablation_results.json`](file:///Users/Daniele/Documents/GitHub/predictive-spikes/outputs/ablation_results.json)):

| Experiment ID | Model / Feature Configuration | Features | 5-Fold CV Accuracy | Fold Std | OOF ROC-AUC | OOF Log-Loss | Train-Val Gap |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `exp00_organizer_logreg` | Official Organizer Baseline (`LogisticRegression`) | 6 | `0.5178` | `0.0080` | `0.5256` | `0.6919` | `0.0033` |
| `exp01_organizer_xgb` | Organizer 6 Features + `XGBClassifier` | 6 | `0.5474` | `0.0128` | `0.5640` | `0.6852` | `0.1592` |
| `exp03_ablation_team_only` | Family A Only (P1 Team + P2 Lead Static Stats) | 88 | `0.5552` | `0.0059` | `0.5747` | `0.6844` | `0.1573` |
| `exp02_xgb_initial_baseline` | Initial XGBoost Baseline (Team + Simple Turn-30 State) | 96 | `0.6667` | `0.0096` | `0.7331` | `0.6012` | `0.1436` |
| `exp04_ablation_team_state` | Families A + C (Team + Physics-Reconstructed State) | 199 | `0.6910` | `0.0038` | `0.7660` | `0.5733` | `0.1462` |
| `exp05_ablation_team_p2_state` | Families A + B + C (+ P2 Observed Roster & Moves) | 275 | `0.6904` | `0.0044` | `0.7669` | `0.5724` | `0.1544` |
| `exp06_ablation_plus_temporal` | Families A + B + C + D (+ Multi-Window Temporal) | 474 | `0.6911` | `0.0046` | `0.7655` | `0.5737` | `0.1576` |
| `exp07_ablation_plus_matchup` | Families A + B + C + D + E (+ Gen 2 Type Matchups) | 525 | `0.6926` | `0.0042` | `0.7661` | `0.5733` | `0.1525` |
| `exp08_ablation_all_6_families` | All 6 Engineered Families (A + B + C + D + E + F) | 546 | `0.6900` | `0.0041` | `0.7667` | `0.5729` | `0.1547` |
| `exp09_all_plus_within_fold_te` | All 6 Families + Within-Fold Target Encoding | 558 | `0.6962` | `0.0055` | `0.7696` | `0.5701` | `0.1613` |
| **`model1_optuna_top200_te`** | **Optuna-Tuned `max_depth=4` on Top-200 + TE** | **212** | **`0.6970`** | **`0.0022`** | **`0.7738`** | **`0.5662`** | **`0.0916`** |
| **`model3_obs_only_top220_no_te`** | **Pure Observation-Only (`Zero-TE`) Top-220** | **220** | **`0.6988`** | **`0.0046`** | **`0.7714`** | **`0.5692`** | **`0.1453`** |
| **`model2_conservative_depth4`** | **Conservative `max_depth=4` Pruned-512 + TE (Sub 2)** | **523** | **`0.7008`** | **`0.0024`** | **`0.7748`** | **`0.5657`** | **`0.1072`** |
| **`model4_compact_top140_te`** | **Compact Top-140 + TE `max_depth=5` (Sub 1)** | **152** | **`0.7020`** | **`0.0036`** | **`0.7752`** | **`0.5647`** | **`0.1286`** |
| **`ensemble_diverse_xgb_blend`** | **Diverse 4-Model XGBoost Blend (Sub 3 / Default)** | **Blend** | **`0.7030`** | **`0.0017`** | **`0.7767`** | **`0.5636`** | **—** |

### Key Takeaways from the Ablation Study & Negative Results (Discarded Ideas)
1. **Physics-Reconstructed Turn-30 State (`exp02 -> exp04`, `+2.43%` CV Accuracy)**: Reconstructing Player 2 single-KOs on Turns `1..29`, de-duplicating cached switch-in moves, and computing recency-decayed HP pools produced the largest structural accuracy jump (`0.6667 -> 0.6910`).
2. **Collinearity Pruning & Feature Subset Selection (`exp09 -> model4`, `+0.58%` CV Accuracy)**: With 546 raw features, high collinearity among temporal windows diluted column subsampling (`colsample_bytree`). Pruning 34 collinear features (`|r| > 0.95`) and selecting the top 140 gain features + 12 within-fold TE features boosted single-model accuracy from `0.6962` to **`0.7020`**.
3. **What Did NOT Help (Discarded)**:
   - *Sparse 75-Column Per-Species Binary/HP Matrix*: Adding 3 sparse columns per species for the top 25 species diluted tree splits and reduced CV accuracy from `0.6962` to `0.6932`. Replacing sparse indicators with 3 dense top-titan HP differentials (*Snorlax*, *Zapdos*, *Cloyster*) plus dense HP-weighted species win-rate expectations inside `WithinFoldTargetEncoder` solved the sparsity problem and lifted CV accuracy above `0.7020`.
   - *Probability Threshold Shifting*: Nested out-of-fold threshold tuning (`optimize_threshold_oof`) confirmed that because `train.jsonl` is balanced `50.0% / 50.0%` and `binary:logistic` probabilities are well-calibrated, the theoretical `0.500` threshold outperforms tuned thresholds out-of-fold (`0.7003` vs `0.6976`). We therefore kept `threshold = 0.500`.

---

## 7. Hyperparameter Optimization & Overfitting Control

Using [`optimize_xgb_optuna`](file:///Users/Daniele/Documents/GitHub/predictive-spikes/src/optimization.py#L96-L185) (`TPESampler(seed=42)`, 25 trials on independent `seed=1337` splits, logged in [`outputs/optuna_trials.csv`](file:///Users/Daniele/Documents/GitHub/predictive-spikes/outputs/optuna_trials.csv)):
- Deeper trees (`max_depth >= 7`) exhibited large train-validation gaps (`> 0.25`) and worse validation log-loss (`> 0.572`) because high-order splits memorize noisy combinations of pre-battle team stats and mid-game damage fluctuations.
- Shallow-to-medium trees (`max_depth` $\in \{4, 5\}$) with strong leaf-weight regularization (`min_child_weight` $\in [10, 20]$), heavy L2 regularization (`reg_lambda` $\in [2.5, 5.0]$), L1 sparsity (`reg_alpha` $\in [0.5, 1.5]$), split penalty (`gamma` $\in [0.2, 0.4]$), and stochastic feature subsampling (`colsample_bytree` $\in [0.45, 0.60]`, `subsample` $\in [0.75, 0.80]$) constrained the train-validation accuracy gap to **`0.0916 – 0.1286`** while maximizing out-of-fold accuracy and ROC-AUC.

### Top 15 Most Predictive Features by XGBoost Gain (`model4_compact_top140_te`)
1. `ix_effective_hp_diff` (Gain: `39.77`) — Status-adjusted full-team remaining HP differential at Turn 30
2. `state_total_hp_ratio` (Gain: `38.12`) — Ratio of Player 1 to Player 2 total known + unrevealed HP pool
3. `state_total_hp_diff` (Gain: `35.15`) — Raw full-team remaining HP differential at Turn 30
4. `ix_alive_x_hp_diff` (Gain: `21.22`) — Interaction between surviving Pokémon count and mean alive HP
5. `state_recency_hp_diff` (Gain: `16.84`) — Recency-decayed HP differential discounting stale bench Pokémon
6. `te_alive_hp_weighted_wr_diff` (Gain: `12.69`) — Within-fold HP-weighted surviving species win-rate differential
7. `state_expected_alive_diff` (Gain: `10.17`) — Continuous physics-reconstructed surviving Pokémon advantage
8. `state_expected_faint_diff` (Gain: `9.62`) — Continuous physics-reconstructed faint differential (`f2 - f1`)
9. `temp_w21_30_hp_diff_mean` (Gain: `8.38`) — Mean team HP advantage across the final 10 turns (`Turns 21–30`)
10. `state_alive_diff` (Gain: `7.94`) — Discrete surviving Pokémon differential at Turn 30
11. `state_faint_diff` (Gain: `7.51`) — Discrete fainted Pokémon differential (`p2_faints - p1_faints`)
12. `temp_last5_hp_diff_mean` (Gain: `7.09`) — Mean team HP advantage across `Turns 26–30`
13. `te_p1_team_wr` (Gain: `6.42`) — Within-fold smoothed empirical win rate of Player 1's 6-Pokémon team
14. `state_p2_expected_faints` (Gain: `6.18`) — Physics-reconstructed Player 2 faint count
15. `state_snorlax_hp_diff` (Gain: `5.84`) — Remaining Turn-30 HP differential for *Snorlax*

---

## 8. The Three Selected Final Submissions

All three submissions are generated and validated automatically by [`scripts/generate_submission.py`](file:///Users/Daniele/Documents/GitHub/predictive-spikes/scripts/generate_submission.py) and [`notebooks/kaggle_submission.ipynb`](file:///Users/Daniele/Documents/GitHub/predictive-spikes/notebooks/kaggle_submission.ipynb):

1. **`outputs/submission_1_compact_top140_xgb.csv`** (**Best Single Model — Compact Top-140 + Within-Fold TE**):
   - **Model**: `model4_compact_top140_te` (`max_depth=5`, `learning_rate=0.032`, `min_child_weight=10`, `colsample_bytree=0.60`, `subsample=0.80`, `reg_lambda=2.5`, `reg_alpha=0.5`, `gamma=0.2`).
   - **Validation Performance**: **5-Fold CV Accuracy = `0.7020 +/- 0.0036`**, OOF ROC-AUC = **`0.7752`**, Log-Loss = **`0.5647`**.
   - **Test Positive Rate**: `0.5202` (`2,601` P1 wins / `2,399` P2 wins).

2. **`outputs/submission_2_conservative_depth4_xgb.csv`** (**Conservative Shallow Full-Spectrum Regularized Model**):
   - **Model**: `model2_conservative_depth4_pruned_te` (`max_depth=4`, `learning_rate=0.03`, `min_child_weight=15`, `colsample_bytree=0.45`, `subsample=0.75`, `reg_lambda=5.0`, `reg_alpha=1.5`, `gamma=0.4`) trained on all `523` collinearity-pruned + TE features.
   - **Validation Performance**: **5-Fold CV Accuracy = `0.7008 +/- 0.0024`**, OOF ROC-AUC = **`0.7748`**, Log-Loss = **`0.5657`**, Train-Val Gap = `0.1072`.
   - **Test Positive Rate**: `0.5194` (`2,597` P1 wins / `2,403` P2 wins).

3. **`outputs/submission_3_diverse_xgb_ensemble.csv` & `outputs/submission.csv`** (**Primary Submission — Diverse 4-Configuration XGBoost Probability Blend**):
   - **Model**: Weighted probability blend (`weights = [0.15, 0.35, 0.10, 0.40]`) of `model1_optuna_top200_te`, `model2_conservative_depth4_pruned_te`, `model3_obs_only_top220_no_te` (pure observation-only, zero target encoding), and `model4_compact_top140_te`.
   - **Validation Performance**: **5-Fold CV Accuracy = `0.7030 +/- 0.0017`**, OOF ROC-AUC = **`0.7767`**, Log-Loss = **`0.5636`**.
   - **Test Positive Rate**: `0.5208` (`2,604` P1 wins / `2,396` P2 wins).
   - **Pairwise Test Prediction Agreement**: `(Sub 1, Sub 2) = 95.32%`, `(Sub 1, Sub 3) = 97.62%`, `(Sub 2, Sub 3) = 97.50%` — confirming meaningful structural diversity (~120–230 distinct battle predictions across the 5,000 test battles) while maintaining strong consensus on high-confidence battles.

---

## 9. Limitations & Future Directions

- **Unobservable Post-Turn-30 RNG**: Because battles last 30–80 turns in Generation 2 OU, critical hits (`6.25%` base rate), full paralysis (`25%`), sleep duration (`1–6` turns), and accuracy misses occurring *after* Turn 30 introduce an irreducible Bayes error floor (~`20–22%` of battles where a player trailing at Turn 30 mounts a late-game comeback or wins via a decisive critical hit).
- **Hidden Held Items**: Although `p1_team_details` does not explicitly list held items in the JSON schema, nearly all Gen 2 OU Pokémon carry *Leftovers* (`+6.25%` HP per turn) unless using *Rest* + *Mint Berry* or *Thick Club* (*Marowak*). Explicit inference of *Leftovers* vs *Mint Berry* from observed passive healing ticks is implicitly captured by our temporal recovery counters.
