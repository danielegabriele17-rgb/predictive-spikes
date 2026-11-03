# Pokémon Battle Prediction 2026 — Competitive XGBoost Pipeline

End-to-end, reproducible, competition-compliant **XGBoost** pipeline for Sapienza University's *Foundations of Data Science — Pokémon Battle Prediction 2026* Kaggle competition.

Given the complete 6-Pokémon roster of Player 1 (`p1_team_details`), the starting lead of Player 2 (`p2_lead_details`), and the first **30 active turns** of a Generation 2 OU battle (`battle_timeline`), our pipeline predicts the final match winner (`player_won` $\in \{0, 1\}$).

---

## Key Results Summary (5-Fold Stratified Cross-Validation, `seed=42`)

| Experiment / Model | Features | 5-Fold CV Accuracy | Fold Std | OOF ROC-AUC | OOF Log-Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Official Organizer Baseline (`LogisticRegression`)** | 6 | `0.5178` | `0.0080` | `0.5256` | `0.6919` |
| **Organizer 6 Features + `XGBClassifier`** | 6 | `0.5474` | `0.0128` | `0.5640` | `0.6852` |
| **Family A Only (Static P1 Team + P2 Lead)** | 88 | `0.5552` | `0.0059` | `0.5747` | `0.6844` |
| **Initial XGBoost Baseline (Team + Simple State)** | 96 | `0.6667` | `0.0096` | `0.7331` | `0.6012` |
| **Families A + C (Team + Physics-Reconstructed State)** | 199 | `0.6910` | `0.0038` | `0.7660` | `0.5733` |
| **All 6 Families + Within-Fold Target Encoding** | 558 | `0.6962` | `0.0055` | `0.7696` | `0.5701` |
| **`model1_optuna_top200_te` (Optuna-Tuned `max_depth=4`)** | 212 | `0.6970` | `0.0022` | `0.7738` | `0.5662` |
| **`model3_obs_only_top220_no_te` (Pure Observation-Only, Zero-TE)** | 220 | `0.6988` | `0.0046` | `0.7714` | `0.5692` |
| **Submission 2: `model2_conservative_depth4_pruned_te`** | 523 | **`0.7008`** | **`0.0024`** | **`0.7748`** | **`0.5657`** |
| **Submission 1: `model4_compact_top140_te`** | 152 | **`0.7020`** | **`0.0036`** | **`0.7752`** | **`0.5647`** |
| **Submission 3: Diverse 4-Model XGBoost Blend (`submission.csv`)** | Blend | **`0.7030`** | **`0.0017`** | **`0.7767`** | **`0.5636`** |

---

## Repository Structure

```text
predictive-spikes/
├── README.md                          # Project overview, results, and execution guide
├── requirements.txt                   # Reproducible Python dependencies
├── .gitignore                         # Excludes fds-pokemon-battles-prediction-2026/, outputs/, .venv/
├── notebook baseline.ipynb            # Official organizer baseline notebook (preserved unchanged)
├── configs/
│   └── experiments.yaml               # Saved feature subsets, tuned XGBoost params, and ensemble weights
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Local & Kaggle (/kaggle/input) environment-aware dataset discovery
│   ├── battle_features.py             # Single-pass 30-turn physics state reconstruction & 6 feature families
│   ├── feature_engineering.py         # Feature matrix builder & leak-free WithinFoldTargetEncoder
│   ├── validation.py                  # StratifiedKFold / duplicate-safe StratifiedGroupKFold & OOF metrics
│   ├── model.py                       # XGBClassifier factory, 5-fold CV runner, feature importance
│   ├── optimization.py                # Collinearity pruning, Optuna TPE tuning, OOF ensemble evaluation
│   └── submission.py                  # Competition CSV generator & strict schema validator
├── scripts/
│   ├── evaluate.py                    # Runs baseline comparisons & progressive feature family ablations
│   ├── train.py                       # Runs collinearity pruning, Optuna search, and saves configs/experiments.yaml
│   └── generate_submission.py         # Reproduces the 3 final validated submission CSV files from configs/
├── notebooks/
│   └── kaggle_submission.ipynb        # Self-contained Kaggle notebook that clones repo, trains, & writes 3 submissions
├── reports/
│   └── methodology_report.md          # Comprehensive technical & academic methodology report
└── tests/
    └── test_pipeline.py               # Automated pytest suite verifying schema, physics parser, and zero leakage
```

---

## Core Technical Highlights

1. **Stateful `move_details` De-duplication**: Once a Pokémon uses a move on turn $t_0$, the competition simulator parser caches `move_details` on that Pokémon object across subsequent switch-outs. When that Pokémon voluntarily switches back in on a later turn $t > t_0$, `move_details` repeats its cached move from $t_0$. Our parser (`reconstruct_battle_history` in `src/battle_features.py`) filters out voluntary switch-in turns (`name[t] != name[t-1]` when `status[t-1] != "fnt"`) so cached moves are never double-counted.
2. **Player 2 Single-KO Physics Reconstruction**: Across 300,000 training turns, `p1_pokemon_state.status == "fnt"` occurs `16,882` times, whereas `p2_pokemon_state.status == "fnt"` occurs only `5,040` times (`98.2%` on simultaneous double-KOs and `1.8%` on Turn 30; **zero** single-KOs on Turns 1–29 because P2's replacement Pokémon immediately appears in the turn snapshot). Reconstructing P2 single-KOs using Generation 2 physics increases the Pearson correlation of material differential (`f2 - f1`) with `player_won` from `+0.2557` to `+0.3360`.
3. **Within-Fold Species Catalogue & Leak-Free Empirical-Bayes Target Encoding**: Builds an internal species lookup strictly from `p1_team_details` and `p2_lead_details` (`247` species with `0` stat/type inconsistencies) and computes smoothed out-of-fold win-rate expectations (`WithinFoldTargetEncoder`) strictly inside each cross-validation fold.

---

## Reproducibility & Local Execution Guide

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Automated Unit & Leakage Tests
```bash
pytest tests/test_pipeline.py -v
```

### 3. Reproduce Progressive Feature Ablation Study (`outputs/ablation_results.json`)
```bash
python scripts/evaluate.py
```

### 4. Run Full Training & Optuna Hyperparameter Optimization (`configs/experiments.yaml`)
```bash
python scripts/train.py
```

### 5. Reproduce the Three Final Competition Submissions (`outputs/*.csv`)
```bash
python scripts/generate_submission.py
```
This generates and validates:
- `outputs/submission_1_compact_top140_xgb.csv` (`CV Acc = 0.7020 +/- 0.0036`, `ROC-AUC = 0.7752`)
- `outputs/submission_2_conservative_depth4_xgb.csv` (`CV Acc = 0.7008 +/- 0.0024`, `ROC-AUC = 0.7748`)
- `outputs/submission_3_diverse_xgb_ensemble.csv` (`CV Acc = 0.7030 +/- 0.0017`, `ROC-AUC = 0.7767`)
- `outputs/submission.csv` (Primary submission, identical to Submission 3)

---

## Kaggle Execution (`notebooks/kaggle_submission.ipynb`)

Upload [`notebooks/kaggle_submission.ipynb`](file:///Users/Daniele/Documents/GitHub/predictive-spikes/notebooks/kaggle_submission.ipynb) to Kaggle Notebooks with the competition dataset attached under `/kaggle/input/`. The notebook automatically:
1. Clones the GitHub repository (`https://github.com/danielegabriele17-rgb/predictive-spikes.git`, or uses the local workspace if already present).
2. Discovers `/kaggle/input/...` via `src.data_loader.discover_data_dir()`.
3. Extracts all 6 feature families and within-fold target encodings.
4. Trains the selected models from `configs/experiments.yaml` across 5 stratified folds.
5. Writes and validates all three final submission CSV files in `/kaggle/working/` with detailed Markdown methodology explanations before each submission.
