# YantraGuard AI — Demo Checklist
*Predictive Maintenance & Intelligent Machine Health Decision System*

## Pre-Demo Setup

- [ ] Python 3.10+ installed
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Dataset present: `data/raw/predictive_maintenance.csv`
- [ ] Model trained: `python -m src.train` (produces `data/processed/best_model.joblib`)
- [ ] `.env` populated with IBM credentials (optional — dashboard works in local mode without them)

---

## Demo Script

### 1. Start the Dashboard
```bash
streamlit run dashboard/app.py
```
Opens at http://localhost:8501

### 2. Overview Page
- [ ] YantraGuard AI logo visible in sidebar
- [ ] Tagline: "Predict Before Failure. Protect Before Downtime."
- [ ] 4 metric cards: Dataset, Training Samples, Test PR-AUC, Test Recall
- [ ] 10-step workflow diagram visible
- [ ] IBM Cloud architecture displayed
- [ ] IBM connection status shown in sidebar

### 3. Machine Health Prediction — Normal Operating Conditions
- Set: Type=M, Air Temp=298.1K, Process Temp=308.6K, RPM=1551, Torque=42.8Nm, Wear=0min
- Click **Predict Machine Health**
- [ ] Probability gauge shows LOW (green)
- [ ] Risk badge: LOW
- [ ] Priority: Routine or Monitor
- [ ] Source shown (IBM Endpoint or Local Model)

### 4. Machine Health Prediction — High-Stress Conditions
- Set: Type=L, Air Temp=302K, Process Temp=311K, RPM=1380, Torque=60Nm, Wear=210min
- Click **Predict Machine Health**
- [ ] Probability gauge shows CRITICAL (purple)
- [ ] Risk badge: CRITICAL
- [ ] Priority: Immediate
- [ ] Recommendation: "Immediate shutdown and inspection"

### 5. Risk Drivers / Explanation
- After high-stress prediction, navigate to **Risk Drivers / Explanation**
- [ ] SHAP bar chart visible
- [ ] Top-5 contributing features shown
- [ ] Red bars = increases risk, green bars = decreases risk
- [ ] Disclaimer prominently displayed

### 6. Inspection Priority
- [ ] Priority gauge shows Immediate (purple)
- [ ] Score breakdown (risk_norm, tool_wear_norm, torque_norm) visible
- [ ] Rule-based disclaimer visible

### 7. Maintenance Recommendation
- [ ] Urgency banner: CRITICAL — Act now
- [ ] Action: "Immediate shutdown and inspection"
- [ ] Sensor notes: Tool wear note (230 min → check cutting tool)
- [ ] Caveats displayed

### 8. What-if Simulator
- Start with high-stress baseline
- Reduce Tool Wear to 10, Torque to 30
- Click **Simulate Modified Scenario**
- [ ] Modified probability drops significantly
- [ ] Delta % shown (should be negative/green)
- [ ] Side-by-side comparison visible

### 9. Model Information
- [ ] Selected model: Random Forest
- [ ] Threshold: 0.713
- [ ] Validation comparison table (3 models)
- [ ] Test set metrics: PR-AUC=0.9411, Recall=0.9020
- [ ] Confusion matrix heatmap

### 10. Dataset / Validation Summary
- [ ] 10,000 rows, 10 columns, 0 missing, 0 duplicates
- [ ] Failure rate: 3.39%, imbalance 28.5:1
- [ ] Failure type pie/bar chart
- [ ] Numeric statistics table
- [ ] Validation: PASSED

---

## Test Suite Verification
```bash
pytest tests/ -v
```
Expected: **101 passed** in ~2 seconds

---

## Secrets Audit
```bash
python secrets_check.py
```
Expected: No hardcoded secrets found. PASSED.

---

## 35-Point Requirements Checklist

| # | Requirement | Status |
|---|-------------|--------|
| 1 | Project name: YantraGuard AI | DONE |
| 2 | Logo at assets/yantraguard-logo.png | DONE |
| 3 | Dark industrial branding | DONE |
| 4 | AI4I 2020 dataset | DONE |
| 5 | Data validation report | DONE (actual results) |
| 6 | Target: Machine failure (binary) | DONE (`Target` column) |
| 7 | Leakage prevention | DONE (Failure Type & Target excluded) |
| 8 | Model input features | DONE (6 operational features) |
| 9 | Feature engineering | DONE (3 derived features inside Pipeline) |
| 10 | Class imbalance analysis | DONE (28.5:1, class_weight=balanced) |
| 11 | Train/val/test strategy | DONE (70/15/15 stratified, seed=42) |
| 12 | Logistic Regression baseline | DONE |
| 13 | Random Forest | DONE (selected as best) |
| 14 | XGBoost | DONE |
| 15 | Fair model comparison | DONE (actual metrics table) |
| 16 | Model selection by failure recall | DONE (PR-AUC + F2) |
| 17 | Threshold optimisation | DONE (0.713, F2 on val set) |
| 18 | Calibration check | DONE (ECE=0.0331) |
| 19 | SHAP explainability | DONE |
| 20 | Risk engine | DONE (4 levels, configurable) |
| 21 | Priority engine | DONE (rule-based, clearly labelled) |
| 22 | Maintenance recommendation | DONE (rule-based, decision support) |
| 23 | What-if simulator | DONE |
| 24 | What-if safety labels | DONE |
| 25 | Streamlit dashboard, 8 sections | DONE |
| 26 | Input controls + validation | DONE |
| 27 | Full output display | DONE |
| 28 | IBM Cloud Object Storage | DONE (auto via watsonx.ai Studio) |
| 29 | watsonx.ai Studio notebook | DONE |
| 30 | watsonx.ai Runtime | DONE (terminology + client) |
| 31 | Online model deployment | DONE (client.deploy_model()) |
| 32 | Dashboard ↔ IBM endpoint | DONE (ibm_client.score()) |
| 33 | Security — no hardcoded secrets | DONE (env vars only) |
| 34 | Tests + documentation | DONE (101 tests, full docs/) |
| 35 | End-to-end demonstration | DONE |
