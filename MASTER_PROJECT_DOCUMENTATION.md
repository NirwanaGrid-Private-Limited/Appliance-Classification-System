# NirwanaGrid — Appliance Classification System

## Master Project Documentation

**Project:** Appliance Classification System  
**Organization:** NirwanaGrid  
**Document Version:** 2.0.0  
**Date:** 2026-07-30  
**Status:** Internship Complete — Final Handover

---

## 1. Project Background

In modern smart grid and residential energy monitoring systems, raw sensor hardware (PZEM power meters) captures real-time electrical measurements (voltage, current, active power, power factor, and frequency) at high polling rates (~6–7 seconds). Individual instantaneous readings can be noisy and ambiguous across different household appliances with overlapping electrical profiles. This project builds a production-ready Machine Learning system that classifies which household appliance is running from live electrical sensor streams, using one-minute aggregated windows to eliminate single-sample noise.

---

## 2. Internship Objective

The assigned internship task was to **extend the existing 4-class appliance classifier by successfully integrating a fifth appliance class (Heater)** into the production ML pipeline, while preserving the existing architecture, feature engineering, and production workflow.

Specifically, the objectives were:

1. Integrate the new Heater raw CSV dataset into the existing data pipeline
2. Update preprocessing to handle the new appliance label
3. Verify feature engineering compatibility (same 15-feature vector)
4. Retrain the production LightGBM model to classify 5 classes
5. Validate the retrained model's performance
6. Ensure backward compatibility with the existing 4 appliance classes (AC, Geyser, LED, Refrigerator)
7. Produce production-ready documentation and report all observations

**This objective was successfully completed.**

---

## 3. Existing System Overview

### Production Model (Pre-Internship)
- **Model:** `LightGBM_v4_combined`
- **Classes:** AC, Electric Geyser, LED, Refrigerator (4 classes)
- **Macro F1:** 0.9708
- **Feature Vector:** 15 features (6 raw electrical + 5 temporal cyclical + 4 engineered electrical)
- **Validation Strategy:** Stratified 80/20 Split
- **Inference Latency:** < 15 ms

### Production Architecture
```
[ PZEM Live Sensor Node API ] → HTTP GET (Bearer Token Auth) every 10s
              ↓
[ LiveCollector Daemon Thread ] (src/collector.py)
              ↓
[ RollingWindow Buffer ] (src/rolling_window.py) — 60s sliding window
              ↓
[ WindowAggregator ] (src/aggregator.py) — 1-minute mean/mode
              ↓
[ FeaturePipeline ] (src/feature_pipeline.py) — 15 features
              ↓
[ Predictor + ModelLoader Singleton ] (src/predictor.py)
              ↓
[ FastAPI Production Service ] (src/app.py) — GET /health, POST /predict
```

### Model Evolution (Pre-Internship)

| Version | Features | Macro F1 | Decision |
|---------|----------|----------|----------|
| LightGBM_v1 | 6 raw | 0.9416 | Baseline |
| LightGBM_v2_time | 11 (raw + temporal) | — | Intermediate |
| LightGBM_v3_electrical | 10 (raw + electrical) | 0.9433 | Intermediate |
| **LightGBM_v4_combined** | **15 (all)** | **0.9708** | **Production** |

---

## 4. Dataset Overview

### Original Dataset (v1)
- **Source:** 9 raw CSV files across multiple sensor nodes
- **Size:** 23,835 cleaned samples in `cleaned_dataset.csv`
- **Classes:** AC, Geyser, LED, Refrigerator

### Extended Dataset (v2 — After Heater Integration)
- **New file added:** `Heater_NODE_40.csv` (3,038 samples)
- **Size:** 26,873 cleaned samples in `cleaned_dataset_v2.csv`
- **Classes:** AC, Geyser, Heater, LED, Refrigerator

| Appliance | Samples | Percentage | Source Nodes |
|-----------|---------|------------|-------------|
| AC | 11,387 | 42.37% | NODE_17, NODE_21, NODE_4 |
| Refrigerator | 9,632 | 35.84% | NODE_3 |
| Heater | 3,038 | 11.31% | NODE_40 |
| Geyser | 1,814 | 6.75% | NODE_4, NODE_18 |
| LED | 1,002 | 3.73% | NODE_3 |

---

## 5. Data Engineering

### Cleaning Pipeline (`src/cleaner.py`, `src/merger.py`)
1. **Schema Validation:** Every CSV validated against `config.EXPECTED_COLUMNS`
2. **Out-of-Bounds Filtering:** V ∈ [180, 260]V, I ∈ [0, 100]A, P ∈ [0, 15000]W, PF ∈ [0, 1.0], F ∈ [45, 55]Hz
3. **Deduplication:** Removed exact duplicate timestamps
4. **Missing Value Handling:** Dropped incomplete frames
5. **Leakage Prevention:** Removed non-causal `source_file` column; preserved `nodeid` for group-level validation

### Validation Pipeline (`src/validator.py`)
- Flags out-of-bounds readings, duplicate timestamps, and missing values
- Invalid rows logged to `data/processed/invalid_rows.csv`

---

## 6. Feature Engineering

The `FeaturePipeline` (`src/feature_pipeline.py`) generates an exact 15-element feature vector:

| Category | Features | Description |
|----------|----------|-------------|
| Raw Electrical (6) | `voltage`, `current`, `power`, `pf`, `frequency`, `relay` | Direct sensor readings (1-min aggregated mean/mode) |
| Temporal Cyclical (5) | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `is_weekend` | Cyclical time-of-day and day-of-week encodings |
| Engineered Electrical (4) | `apparent_power`, `reactive_power`, `current_pf`, `power_per_voltage` | V×I, √(S²−P²), I×PF, P/V |

This exact feature vector is shared between training, batch evaluation, and live inference to ensure zero training-serving skew.

---

## 7. Baseline Production Model (`LightGBM_v4_combined`)

| Metric | Value |
|--------|-------|
| Macro F1 | 0.9708 |
| Macro Precision | 0.9633 |
| Macro Recall | 0.9793 |
| AC Recall | 0.9991 |
| Geyser Recall | 0.9945 |
| LED Recall | 0.9400 |
| Refrigerator Recall | 0.9834 |
| Hyperparameters | Default LightGBM |
| Validation | Stratified 80/20, random_state=42 |

---

## 8. Heater Integration (Internship Core Deliverable)

### 8.1 Data Integration
- **Script:** `src/integrate_heater_data.py`
- **Input:** `Heater_NODE_40.csv` from `data/raw/`
- **Process:** Schema validation → cleaning → label assignment → append to master dataset
- **Output:** `data/processed/cleaned_dataset_v2.csv` (26,873 rows, 5 classes)
- **Label Mapping:** Updated `src/data_loader.py` with extensible dictionary-based mapping supporting the new `Heater` class

### 8.2 Feature Engineering Compatibility
The Heater integration required **zero changes** to `src/feature_pipeline.py`. The same 15-feature vector was computed for all Heater samples, ensuring full backward compatibility with the existing production pipeline.

### 8.3 Model Retraining (EXP-005)
- **Script:** `src/run_exp_005_heater.py`
- **Model:** `LightGBM_v5_combined`
- **Hyperparameters:** Identical to `LightGBM_v4_combined` (Default LightGBM, random_state=42)
- **Features:** Same 15 features
- **Validation:** Same Stratified 80/20 strategy
- **Only experimental variable:** Addition of the Heater class

### 8.4 Evaluation Results

| Metric | v4 (4 classes) | v5 (5 classes) | Delta |
|--------|----------------|----------------|-------|
| **Macro F1** | 0.9708 | 0.9489 | -0.0219 |
| **Macro Precision** | 0.9633 | 0.9419 | -0.0214 |
| **Macro Recall** | 0.9793 | 0.9566 | -0.0227 |

| Class | v5 Precision | v5 Recall | v5 F1 |
|-------|-------------|-----------|-------|
| AC | 0.9851 | 0.9838 | 0.9844 |
| Geyser | 0.8986 | 0.9036 | 0.9011 |
| **Heater** | **1.0000** | **1.0000** | **1.0000** |
| LED | 0.8356 | 0.9150 | 0.8735 |
| Refrigerator | 0.9900 | 0.9808 | 0.9854 |

### 8.5 Production Compatibility
- **No production code was modified:** FastAPI, model_loader, predictor, inference pipeline, and all deployment assets remain untouched
- **Model artifacts saved:** `LightGBM_v5_combined.joblib` and `LightGBM_v5_combined_metadata.json`
- **Backward compatible:** The v5 model uses the identical 15-feature input vector

**The Heater integration was completed successfully.** The retrained model classifies all 5 appliance classes with a Macro F1 of 0.9489 and achieves perfect classification of the new Heater class.

---

## 9. Validation Results Summary

The retrained `LightGBM_v5_combined` model:
- Achieves **Macro F1 = 0.9489** across 5 classes (vs. 0.9708 for 4 classes)
- Classifies the new **Heater class with 100% Precision and 100% Recall**
- Maintains strong performance on AC (F1=0.9844) and Refrigerator (F1=0.9854)
- Shows moderate degradation on Geyser (F1 dropped from ~0.97 to 0.9011) and LED (F1=0.8735)
- **Zero confusion** between Heater and Geyser (0 misclassifications in either direction)

The Geyser degradation was identified as a secondary effect of introducing the new class into the global tree-building process, not a direct confusion between Heater and Geyser.

---

## 10. Investigation Timeline

Following the successful Heater integration, additional validation experiments were voluntarily conducted to ensure the retrained model's robustness and to investigate the observed Geyser degradation. These experiments were performed as responsible engineering due diligence, separate from the core internship assignment.

### 10.1 Root Cause Analysis (RCA)
- **Script:** `src/run_rca_exp005.py`
- **Objective:** Determine why Geyser recall degraded from 99.5% to 90.4%
- **Method:** Comprehensive analysis spanning dataset distribution, feature separability (PCA/t-SNE), confusion matrix analysis, class-specific SHAP, probability analysis, class imbalance analysis, model capacity analysis, and label quality audit
- **Key Finding:** The confusion matrix showed **zero misclassifications between Heater and Geyser**. All 33 Geyser errors were misclassified as AC. The Node-Level Audit revealed that 100% of Heater data originates from a single node (NODE_40).

### 10.2 EXP-005B: Time Feature Ablation
- **Script:** `src/run_exp_005b_validation.py`
- **Objective:** Test whether the model memorized Heater through temporal features
- **Method:** Retrained v5 with identical settings but removed all 5 temporal features
- **Result:** Heater Recall remained at 100%. Macro F1 dropped by only 0.001 (0.9489 → 0.9479). Temporal features are not the mechanism.
- **Additional test (GroupKFold):** Leave-one-node-out validation showed near-zero cross-node performance for all classes, suggesting the model has difficulty generalizing to completely unseen sensor nodes.

### 10.3 EXP-005C: Node Feature Ablation
- **Script:** `src/run_exp_005c_node_ablation.py`
- **Objective:** Test whether node-specific electrical baselines drive the model's performance
- **Method:** Removed 7 node-baseline features (voltage, frequency, pf, and their derivatives), retained 8 appliance-behavior features
- **Result:** Stratified Macro F1 dropped moderately to 0.9304. Heater still achieved 100% Recall. GroupKFold performance remained near zero, confirming the observation is independent of specific feature subsets.

### 10.4 Dataset Topology Audit
- **Script:** `src/run_topology_audit.py`
- **Objective:** Map the Node × Appliance structure and verify GroupKFold class coverage
- **Key Finding:** 3 of 5 appliance classes (Heater, LED, Refrigerator) appear on only 1 node each. In GroupKFold, 2 of 6 folds tested classes with zero training examples (expected failure), while 4 of 6 folds showed genuine cross-node generalization limitations.

### Experiment Archive

The following intermediate reports were produced during the investigation and have been consolidated into this document:

| Report | Purpose |
|--------|---------|
| `HEATER_INTEGRATION_REPORT.md` | Initial Heater integration evaluation |
| `EXP005_ROOT_CAUSE_ANALYSIS.md` | Root cause analysis of Geyser degradation |
| `EXP005B_VALIDATION_REPORT.md` | Time ablation and GroupKFold validation |
| `EXP005C_NODE_FEATURE_ABLATION_REPORT.md` | Node-baseline ablation and topology audit |
| `class_distribution_v2.md` | Class distribution summary |

---

## 11. Final Technical Findings

### Confirmed by Evidence
1. **The Heater class was successfully integrated** — perfect 100% Recall and Precision under the existing validation methodology
2. **Heater and Geyser do not confuse the model** — zero misclassifications between the two classes in any experiment
3. **The Geyser degradation is caused by structural tree shifts**, not by electrical overlap with Heater
4. **Temporal features contribute negligibly** — removing all 5 changed Macro F1 by only 0.001
5. **3 of 5 appliance classes are present on only 1 sensor node**, limiting cross-node evaluation

### Partially Supported
1. **Cross-node generalization is limited** — confirmed for the 4 valid GroupKFold folds, but 2 folds were structurally impossible (missing training classes)

### Cannot Be Concluded
1. Whether collecting additional nodes would fully resolve cross-node generalization — this requires future experimentation with expanded data
2. Whether the existing production model (v4) exhibits the same cross-node limitations — this was not tested

---

## 12. Final Engineering Observations

**The Heater integration itself was successful.** The retrained model performs well under the current validation strategy (Stratified 80/20), achieving Macro F1 = 0.9489 with perfect Heater classification and maintaining strong performance on the existing classes.

**Additional experiments revealed observations** related to dataset topology and node diversity. Specifically, several appliance classes are represented by data from only 1–2 sensor nodes (households), and cross-node evaluation using GroupKFold shows limited generalization. These observations are important for future development — particularly if the system is to be deployed across diverse households — but they **do not invalidate the successful completion of the Heater integration task**.

The production inference pipeline (FastAPI, model_loader, predictor) was deliberately left untouched throughout this work, preserving full backward compatibility.

---

## 13. Deliverables Produced

### Core Deliverables (Internship Assignment)
| Deliverable | Status |
|-------------|--------|
| Heater dataset integrated into pipeline | ✅ Complete |
| Label mapping updated (`src/data_loader.py`) | ✅ Complete |
| Merged dataset (`cleaned_dataset_v2.csv`) | ✅ Complete |
| Retrained model (`LightGBM_v5_combined.joblib`) | ✅ Complete |
| Model metadata (`LightGBM_v5_combined_metadata.json`) | ✅ Complete |
| Training script (`src/run_exp_005_heater.py`) | ✅ Complete |
| Integration script (`src/integrate_heater_data.py`) | ✅ Complete |
| Evaluation report | ✅ Complete |
| Production compatibility verified | ✅ Complete |

### Additional Deliverables (Voluntary Investigation)
| Deliverable | Status |
|-------------|--------|
| Root Cause Analysis script and metrics | ✅ Complete |
| Time Feature Ablation experiment (EXP-005B) | ✅ Complete |
| Node Feature Ablation experiment (EXP-005C) | ✅ Complete |
| Dataset Topology Audit | ✅ Complete |
| Ablation model artifacts (v5B, v5C) | ✅ Complete |
| Investigation plots (PCA, t-SNE, confusion matrices, SHAP) | ✅ Complete |

---

## 14. Repository Structure

```
nirvana_grid/
├── MASTER_PROJECT_DOCUMENTATION.md    # This document — complete project documentation
├── INTERNSHIP_HANDOVER_SUMMARY.md     # Non-technical handover summary for supervisor
├── README.md                          # Quick-start guide
├── requirements.txt                   # Python dependencies
├── .env                               # API token configuration
│
├── data/
│   ├── raw/                           # 10 raw CSV files (including Heater_NODE_40.csv)
│   └── processed/
│       ├── cleaned_dataset.csv        # Original 4-class dataset (v1)
│       ├── cleaned_dataset_v2.csv     # Extended 5-class dataset (v2, with Heater)
│       └── invalid_rows.csv           # Rejected rows from cleaning
│
├── src/
│   ├── config.py                      # Global configuration and schema
│   ├── data_loader.py                 # Dataset loading and label inference
│   ├── validator.py                   # Data validation rules
│   ├── cleaner.py                     # Data cleaning pipeline
│   ├── merger.py                      # Multi-file dataset merger
│   ├── eda.py                         # Exploratory data analysis
│   ├── profiler.py                    # Statistical profiling
│   ├── feature_pipeline.py           # 15-feature engineering pipeline
│   ├── integrate_heater_data.py      # Heater dataset integration script
│   ├── run_exp_005_heater.py         # EXP-005: Heater retraining script
│   ├── run_rca_exp005.py             # Root Cause Analysis diagnostic script
│   ├── run_exp_005b_validation.py    # EXP-005B: Time ablation + GroupKFold
│   ├── run_exp_005c_node_ablation.py # EXP-005C: Node-baseline ablation
│   ├── run_topology_audit.py         # Dataset topology audit
│   ├── run_stage_a.py                # Stage A baseline training
│   ├── run_stage_b.py                # Stage B feature engineering training
│   ├── run_baseline_audit.py         # Baseline audit script
│   ├── run_time_validation.py        # Temporal validation experiments
│   ├── run_live_prediction.py        # Live prediction CLI runner
│   ├── api_client.py                 # PZEM sensor API HTTP client
│   ├── rolling_window.py             # 60-second sliding sensor window
│   ├── aggregator.py                 # Window aggregation (mean/mode)
│   ├── model_loader.py               # Thread-safe singleton model loader
│   ├── predictor.py                  # Core prediction gateway
│   ├── collector.py                  # Background polling daemon
│   ├── app.py                        # FastAPI production service
│   └── pipeline.py                   # Main data pipeline orchestrator
│
├── outputs/
│   ├── models/                        # All model artifacts (.joblib + metadata .json)
│   ├── plots/                         # Visualizations (EDA, confusion matrices, SHAP)
│   └── reports/                       # Experiment metrics (.json) and audit data
│
├── docs/                              # Historical documentation archive
├── logs/                              # Runtime execution logs
└── notebooks/                         # Jupyter notebooks for exploration
```

---

## 15. Deployment Guide

### Environment Configuration
```bash
# .env file in project root
API_TOKEN="your_secure_bearer_token_here"
```

### Installation
```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### Running the Data Pipeline
```bash
python src/pipeline.py
```

### Running the FastAPI Production Service
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --workers 1
```

### API Endpoints
- `GET /health` — System health, buffer state, model readiness
- `POST /predict` — Real-time appliance classification with confidence score

---

## 16. Final Project Status

| Component | Status |
|-----------|--------|
| Data pipeline (cleaning, validation, merging) | ✅ Production ready |
| Feature engineering (15-feature vector) | ✅ Production ready |
| Baseline model (LightGBM_v4_combined, 4 classes) | ✅ Production deployed |
| Heater integration (LightGBM_v5_combined, 5 classes) | ✅ Complete |
| Live inference pipeline | ✅ Production ready |
| FastAPI service | ✅ Production ready |
| Post-integration validation experiments | ✅ Complete |
| Documentation | ✅ Complete |
| Internship handover | ✅ Ready |
