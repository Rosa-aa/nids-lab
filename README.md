# NIDS Lab — Network Intrusion Detection System
 
A data science capstone project applying machine learning to network intrusion detection, trained on the **UNSW-NB15** dataset. The project detects and classifies 9 attack categories (plus Normal traffic) from raw network flow data, and visualizes results through an interactive dashboard.
 
---
 
## Table of Contents
 
1. [Repository Structure & Upload Order](#repository-structure--upload-order)
2. [File-by-File Documentation](#file-by-file-documentation)
   - [notebooks/Final_project.ipynb](#notebooksfinal_projectipynb)
   - [ml_pipeline/](#ml_pipeline)
   - [docs/](#docs)
   - [dashboard/](#dashboard)
3. [Dataset](#dataset)
4. [Results Summary](#results-summary)
5. [Setup & Installation](#setup--installation)
6. [Tech Stack](#tech-stack)
---
 
## Repository Structure & Upload Order
 
Upload/commit in this order — each stage builds on the previous one, so a reviewer scrolling through commit history sees a logical progression from data → model → documentation → visualization.
 
```
nids-lab/
├── README.md                          ← STEP 1 (this file)
├── .gitignore                         ← STEP 1
├── notebooks/
│   └── Final_project.ipynb            ← STEP 2  (EDA + full experimentation)
├── ml_pipeline/
│   ├── README.md                      ← STEP 3
│   ├── requirements.txt               ← STEP 3
│   ├── config.py                      ← STEP 3
│   ├── utils.py                       ← STEP 3
│   ├── data_prep.py                   ← STEP 3
│   ├── feature_engineering.py         ← STEP 3
│   ├── feature_selection.py           ← STEP 3
│   ├── train.py                       ← STEP 3
│   ├── evaluate.py                    ← STEP 3
│   └── run_pipeline.py                ← STEP 3
├── docs/
│   ├── config.md                      ← STEP 4
│   ├── utils.md                       ← STEP 4
│   ├── data_prep.md                   ← STEP 4
│   ├── feature_engineering.md         ← STEP 4
│   ├── feature_selection.md           ← STEP 4
│   ├── train.md                       ← STEP 4
│   ├── evaluate.md                    ← STEP 4
│   └── run_pipeline.md                ← STEP 4
└── dashboard/
    └── nids_dashboard.html            ← STEP 5
```
 
**Why this order matters:**
1. **README + .gitignore first** — sets expectations and prevents accidentally committing large CSVs/`.pkl` files in later steps.
2. **Notebook second** — this is the "research" artifact: all exploration, cleaning decisions, and model comparisons live here. It's the source of truth the `.py` modules were extracted from.
3. **ml_pipeline third** — the production-ready, modular version of the notebook. Depends conceptually on step 2 (same logic, cleaner form).
4. **docs fourth** — detailed, file-by-file documentation of every module in `ml_pipeline/`.
5. **dashboard fifth** — the final presentation layer, consuming results produced by steps 3–4.
### `.gitignore` (create this in Step 1)
```gitignore
# Large data files — never commit these
*.csv
*.parquet
*.pkl
 
# Logs
*.log
 
# OS/editor junk
.DS_Store
.vscode/
__pycache__/
*.pyc
```
 
---
 
## File-by-File Documentation
 
### `notebooks/Final_project.ipynb`
 
**Purpose:** The complete, original Google Colab notebook containing every experimentation step — this is where all decisions were made before being distilled into the clean `ml_pipeline/` modules.
 
**What's inside (in execution order):**
 
| Section | What it does |
|---|---|
| 1. Data Loading | Mounts Google Drive, reads `NUSW-NB15_features.csv` (column names) and the four `UNSW-NB15_1.csv`–`4.csv` raw files, concatenates them into a single DataFrame (~2.06M rows). |
| 2. Data Cleaning | Drops leakage-prone identifier columns (`srcip`, `dstip`, `stcpb`, `dtcpb`), fills missing `attack_cat` with `"Normal"`, standardizes the `"Backdoors"` → `"Backdoor"` label typo, removes duplicate rows, coerces `sport`/`dsport` to numeric and drops invalid rows. |
| 3. Dtype Optimization | Downcasts `float64` → `float32` and `int64` → `int32` across the DataFrame to cut RAM usage roughly in half — necessary given the dataset's size on Colab's free tier. |
| 4. Data Validation | Sanity checks: port range validation (0–65535), negative-value checks on `dur`/`sbytes`/`dbytes`/`sload`/`dload`, and cardinality checks on categorical columns (`proto`, `service`, `state`). |
| 5. Exploratory Data Analysis (EDA) | Visualizations: attack category distribution (log-scale bar), Normal vs Attack balance, protocol/service frequency bars, correlation heatmap of key numeric features, boxplots for outlier inspection, and a pairplot on a 5,000-row sample. |
| 6. Time-Based Train/Test Split | Sorts rows chronologically by `stime` and splits 80/20 — this simulates a realistic deployment scenario where a model is trained on past traffic and evaluated on future traffic, rather than a random shuffle which would leak temporal patterns. |
| 7. Feature Engineering | Creates `bytes_per_pkt` (`sbytes / (spkts + 1)`) and `port_cat` (destination port bucketed into `well_known`/`registered`/`dynamic`). Rare `proto`/`state` values are grouped into `"other"` before one-hot encoding to control dimensionality. |
| 8. Scaling | Applies `RobustScaler` to all numeric columns (robust to the heavy outliers typical of network traffic data). |
| 9. Feature Selection (Mutual Information) | Computes Mutual Information scores between each feature and `attack_cat` on a 50,000-row sample, selects the top 20 features, and checkpoints them to Drive as Parquet/CSV so later cells don't need to repeat the expensive full pipeline. |
| 10. Class Imbalance Handling | Measures the imbalance ratio (Normal vastly outnumbers rare attacks like Worms), then experiments with `SMOTE` (oversampling) + `RandomUnderSampler` as an alternative to `class_weight='balanced'`. |
| 11. Model Training | Trains three models: **Random Forest (class_weight=balanced)**, **Random Forest + SMOTE**, and **XGBoost** — all on the top-20 MI-selected features. |
| 12. Evaluation | `classification_report` (precision/recall/F1 per class), confusion matrix heatmap, and a side-by-side model comparison table (Accuracy + Macro F1). |
| 13. SHAP Explainability | `TreeExplainer` on a stratified sample (equal representation per class) to understand *why* the model misclassifies rare classes — includes a worked example investigating a Backdoor→Normal misclassification via a SHAP force plot. |
 
**Why keep the notebook at all if `ml_pipeline/` exists?** Notebooks are the right tool for *exploration* — showing your reasoning, intermediate plots, and dead ends — which graders/interviewers often want to see. The `.py` modules are the right tool for *reuse and deployment*. Keeping both is standard practice in real data science teams.
 
---
 
### `ml_pipeline/`
 
The notebook's logic refactored into importable, testable Python modules. See [`ml_pipeline/README.md`](ml_pipeline/README.md) for the module table and quick-start commands. Summary of each file's responsibility:
 
- **`config.py`** — Single source of truth for file paths, random seed, column lists, and hyperparameter defaults (e.g. `TOP_PROTO_N`, `N_TOP_FEATURES`). Change a path or constant once here instead of hunting through multiple files.
- **`utils.py`** — Small, dependency-light helpers reused across modules: `port_category()` (port bucketing), `print_section()` (consistent console headers), `check_ram()` (memory monitoring), `free_memory()` (explicit garbage collection helper).
- **`data_prep.py`** — Everything from raw CSV to a clean, split DataFrame: `load_raw_data()`, `clean_data()`, `optimize_dtypes()`, `validate_data()`, `time_based_split()`.
- **`feature_engineering.py`** — `add_engineered_features()` (bytes_per_pkt, port_cat), `encode_categoricals()` (top-N bucketing + one-hot encoding + train/test column alignment), `scale_features()` (RobustScaler).
- **`feature_selection.py`** — `select_top_features_mi()` runs Mutual Information and saves a Parquet/CSV checkpoint; `load_checkpoint()` reloads it without recomputing.
- **`train.py`** — One function per model: `train_random_forest_balanced()`, `train_random_forest_smote()`, `train_xgboost()` (handles the required `LabelEncoder` internally), plus `save_model()` for `joblib` serialization.
- **`evaluate.py`** — `evaluate_model()`, `plot_confusion_matrix()`, `compare_models()`, and optional `shap_summary()` / `shap_stratified_sample()` for explainability.
- **`run_pipeline.py`** — Orchestrates all of the above end-to-end; running `python run_pipeline.py` reproduces the entire notebook pipeline from raw CSVs to a saved model.
---
 
### `docs/`
 
Detailed, standalone documentation for every file in `ml_pipeline/` — each covers the module's purpose, every function's inputs/outputs, the design decisions behind it, and which notebook section it was extracted from.
 
| File | Documents |
|---|---|
| `config.md` | `ml_pipeline/config.py` |
| `utils.md` | `ml_pipeline/utils.py` |
| `data_prep.md` | `ml_pipeline/data_prep.py` |
| `feature_engineering.md` | `ml_pipeline/feature_engineering.py` |
| `feature_selection.md` | `ml_pipeline/feature_selection.py` |
| `train.md` | `ml_pipeline/train.py` |
| `evaluate.md` | `ml_pipeline/evaluate.py` |
| `run_pipeline.md` | `ml_pipeline/run_pipeline.py` |
 
---
 
### `dashboard/`
 
#### `nids_dashboard.html`
**Purpose:** A single, self-contained, offline-capable HTML dashboard (Chart.js is embedded directly in the file — no CDN/internet dependency) built for the thesis defense presentation. Sections:
- **Hero stats** — animated counters for total traffic, threats detected, model accuracy, and macro F1.
- **Attack category distribution** — log-scale bar chart with hover tooltips explaining each of the 10 categories (Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms).
- **Protocol & service analysis** — doughnut/bar charts with hover definitions for `tcp`, `udp`, `arp`, `dns`, `http`, etc.
- **Feature importance** — top-10 Mutual Information scores with plain-language explanations of each feature.
- **Per-class performance table** — Precision/Recall/F1 per class with contextual tooltips (e.g., explaining *why* a class like `Analysis` has low precision but high recall).
- **Model comparison chart** — Random Forest (balanced) vs. Random Forest (SMOTE) vs. XGBoost.
- **Live log ticker** — auto-scrolling simulated detection feed.
All data in this file comes directly from the notebook's real outputs (not placeholder numbers) — see the `const DATA = {...}` block near the top of the `<script>` section if you need to update it with new results.
 
Just open `dashboard/nids_dashboard.html` in any browser — no server required.
 
---
 
## Dataset
 
**UNSW-NB15** — created by the Cyber Range Lab of UNSW Canberra, containing a mix of real modern normal traffic and synthetically generated contemporary attack behaviors across 9 categories.
 
- Official page: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- Kaggle mirror (cleaned/parquet): https://www.kaggle.com/datasets/dhoogla/unswnb15
- Size used in this project: 4 CSV files combined and cleaned to **2,058,693 rows**, 45 raw columns → 20 selected features after Mutual Information ranking.
---
 
## Results Summary
 
Evaluated on a chronological 20% hold-out test set (411,739 rows):
 
| Model | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| **Random Forest (class_weight=balanced)** | 96% | **0.575** | Final model — best macro F1 |
| Random Forest + SMOTE | 96% | 0.566 | Oversampling/undersampling attempted as an alternative |
| XGBoost | 97% | 0.544 | Highest accuracy, but lower macro F1 due to rare-class performance |
 
**Key insight:** Accuracy is misleading here because ~95% of the dataset is Normal traffic. Macro F1 (unweighted average across all 10 classes) is the metric that actually reflects performance on rare-but-critical attack types like `Worms` (56 test samples) and `Shellcode` (494 test samples).
 
---
 
## Setup & Installation
 
### ML Pipeline
```bash
cd ml_pipeline
pip install -r requirements.txt
export UNSW_DATA_DIR=/path/to/UNSW_NB15   # folder containing the 4 raw CSVs
python run_pipeline.py
```
 
### Dashboard
Just open `dashboard/nids_dashboard.html` in any browser — no server required.
 
---
 
## Tech Stack
 
**ML/Data:** Python, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn (SMOTE), SHAP, joblib
**Visualization:** Chart.js, HTML/CSS/JS
**Environment:** Google Colab (training)
 
