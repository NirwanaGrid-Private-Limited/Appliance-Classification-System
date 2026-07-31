# NirwanaGrid — Appliance Classification System

A production-ready Machine Learning system that classifies household appliances from live electrical sensor streams using LightGBM.

## Supported Appliances
- Air Conditioner (AC)
- Electric Geyser
- **Heater** *(newly integrated)*
- LED Lighting
- Refrigerator

## Quick Start

### Installation
```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### Run Data Pipeline
```bash
python src/pipeline.py
```

### Run FastAPI Service
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --workers 1
```

### API Endpoints
- `GET /health` — System diagnostics and readiness status
- `POST /predict` — Real-time appliance classification

## Project Documentation
- **[MASTER_PROJECT_DOCUMENTATION.md](MASTER_PROJECT_DOCUMENTATION.md)** — Complete technical documentation
- **[INTERNSHIP_HANDOVER_SUMMARY.md](INTERNSHIP_HANDOVER_SUMMARY.md)** — Supervisor handover summary

## Repository Structure
```
nirvana_grid/
├── data/raw/                  # Raw CSV sensor data (10 files)
├── data/processed/            # Cleaned datasets (v1 and v2)
├── src/                       # Source code (pipeline, training, inference, experiments)
├── outputs/models/            # Trained model artifacts (.joblib + metadata)
├── outputs/plots/             # Visualizations and experiment plots
├── outputs/reports/           # Experiment metrics (.json)
├── docs/                      # Documentation archive
├── logs/                      # Runtime logs
└── notebooks/                 # Jupyter notebooks
```

## Technology Stack
- **ML Framework:** LightGBM
- **API:** FastAPI + Uvicorn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Validation:** scikit-learn
