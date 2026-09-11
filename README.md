# YantraGuard AI

**Predictive Maintenance & Intelligent Machine Health Decision System**

> *Predict Before Failure. Protect Before Downtime.*

---

## Project Summary

YantraGuard AI is a complete end-to-end predictive maintenance application that:

1. Accepts industrial sensor readings (temperatures, RPM, torque, tool wear, machine type)
2. Predicts failure probability using a Random Forest model deployed on **IBM watsonx.ai Runtime**
3. Converts the probability into an actionable risk level, inspection priority, and maintenance recommendation
4. Explains predictions using SHAP feature contributions
5. Provides a what-if simulator for scenario exploration
6. Presents everything through a professional dark-themed Streamlit dashboard

**Problem Statement:** Mechanical Engineering #39 — Predictive Maintenance of Industrial Machinery  
**Dataset:** AI4I 2020 Predictive Maintenance Dataset (10,000 rows, synthetic benchmark)  
**IBM Cloud region:** us-south (Dallas)

---

## Actual Test Results (from training run)

| Model | Threshold | PR-AUC | ROC-AUC | F2 | Recall | Precision |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.643 | 0.3552 | 0.8854 | 0.4695 | 0.7255 | 0.1947 |
| Random Forest | **0.713** | **0.8332** | **0.9506** | **0.7968** | **0.7843** | **0.8511** |
| XGBoost | 0.337 | 0.8249 | 0.9472 | 0.7905 | 0.7843 | 0.8163 |

**Selected model: Random Forest** (best PR-AUC + F2 on validation set)

### Final Test Set (held-out, n=1500):

| Metric | Value |
|--------|-------|
| Precision | **0.8679** |
| Recall | **0.9020** |
| F1 | **0.8846** |
| F2 (β=2) | **0.8949** |
| ROC-AUC | **0.9828** |
| PR-AUC | **0.9411** |
| TP=46, FP=7, FN=5, TN=1442 | |

Calibration ECE: 0.0331 (no calibration needed)

---

## Project Structure

```
YantraGuard/
├── assets/yantraguard-logo.png      # Official logo
├── data/
│   ├── raw/predictive_maintenance.csv
│   └── processed/                   # Generated artefacts (gitignored)
├── src/
│   ├── config.py                    # Central config — features, thresholds
│   ├── data_validation.py           # Data quality checks
│   ├── preprocessing.py             # sklearn Pipeline factory
│   ├── feature_engineering.py       # FeatureEngineer transformer
│   ├── train.py                     # Training script
│   ├── evaluate.py                  # Metrics, threshold, calibration
│   ├── explainability.py            # SHAP wrapper
│   ├── risk_engine.py               # Probability → RiskLevel
│   ├── priority_engine.py           # Inspection priority (rule-based)
│   ├── recommender.py               # Maintenance recommendation (rule-based)
│   └── ibm_client.py                # IBM watsonx.ai Runtime wrapper
├── dashboard/
│   ├── app.py                       # Streamlit entry point
│   └── pages/                       # 8 page modules
├── tests/                           # 101 pytest tests
├── notebooks/01_yantraguard_training.ipynb  # watsonx.ai Studio notebook
├── docs/                            # All documentation
├── .env.example                     # Credentials template
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env and fill in your IBM Cloud credentials
```

### 3. Train the model (first run)

```bash
python -m src.train
```

This produces:
- `data/processed/best_model.joblib`
- `data/processed/shap_background.joblib`
- `data/processed/training_results.json`
- `data/processed/validation_report.json`

### 4. Run the dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard opens at `http://localhost:8501`

---

## IBM Cloud Setup

1. Create an [IBM Cloud account](https://cloud.ibm.com)
2. Provision **IBM watsonx.ai** (includes Watson Machine Learning + Cloud Object Storage)
3. Create an **API key**: IBM Cloud → Manage → Access → API keys
4. Create a **deployment space** in watsonx.ai Studio → Deployments → New Space
5. Upload `predictive_maintenance.csv` to your watsonx.ai Studio project
6. Run `notebooks/01_yantraguard_training.ipynb` in watsonx.ai Studio
7. Copy the `IBM_DEPLOYMENT_ID` printed at the end of the notebook into `.env`

### Required environment variables

| Variable | Description |
|----------|-------------|
| `IBM_CLOUD_API_KEY` | IBM Cloud IAM API key |
| `IBM_WATSON_ML_INSTANCE_ID` | WML service instance GUID |
| `IBM_WATSON_ML_URL` | `https://us-south.ml.cloud.ibm.com` |
| `IBM_SPACE_ID` | Deployment space GUID |
| `IBM_DEPLOYMENT_ID` | Online deployment GUID (after notebook run) |

---

## Running Tests

```bash
pytest tests/ -v
```

Expected: **101 passed** in ~2 seconds.

IBM client tests use mocks — no live credentials needed.

---

## Limitations

- AI4I 2020 is a **synthetic** benchmark dataset. Results may not transfer to real industrial fleets.
- The model predicts failure probability for individual observations — not fleet-level remaining useful life.
- SHAP values explain the model's decision, not physical root cause.
- All recommendations are decision support only — validate with qualified maintenance personnel.
- IBM deployment requires active IBM Cloud account with watsonx.ai provisioned.
- Without IBM credentials, the dashboard runs in local model fallback mode.

---

## Built With

- Python 3.13 / scikit-learn 1.9 / XGBoost 3.4 / SHAP 0.52
- IBM watsonx.ai Runtime (`ibm-watsonx-ai`)
- Streamlit 1.58 / Plotly 5
- IBM Bob (AI development assistant)
