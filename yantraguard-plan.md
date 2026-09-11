# YantraGuard AI — Implementation Plan
**Predictive Maintenance & Intelligent Machine Health Decision System**
*"Predict Before Failure. Protect Before Downtime."*

---

## Confirmed Decisions (from user)

| # | Question | Decision |
|---|----------|----------|
| 1 | IBM Cloud region | **us-south (Dallas)** — endpoint: `https://us-south.ml.cloud.ibm.com` |
| 2 | Streamlit deployment | **Run locally only** |
| 3 | Dataset | **`predictive_maintenance.csv`** already present in workspace root |
| 4 | Feature engineering toggle | **Always ON** (best-performing configuration used) |
| 5 | SHAP delay | **Acceptable** — cache per session, no extra optimisation needed |

## Dataset Column Verification (Actual)

The supplied CSV (`predictive_maintenance.csv`) uses these **actual column names** (different from AI4I original UCI layout):

| Column | Role |
|--------|------|
| `UDI` | Row identifier — **excluded** |
| `Product ID` | Machine identifier — **excluded** |
| `Type` | Machine quality variant (L/M/H) — **input feature** |
| `Air temperature [K]` | Sensor — **input feature** |
| `Process temperature [K]` | Sensor — **input feature** |
| `Rotational speed [rpm]` | Sensor — **input feature** |
| `Torque [Nm]` | Sensor — **input feature** |
| `Tool wear [min]` | Sensor — **input feature** |
| `Target` | Binary failure label (0/1) — **prediction target** |
| `Failure Type` | Text failure category — **excluded (target-derived)** |

**Note:** `Failure Type` replaces the original TWF/HDF/PWF/OSF/RNF binary columns. It is still target-derived and must be excluded from model inputs.

---

## Top-Level Overview

**Goal:** Build a complete, end-to-end predictive maintenance application using the AI4I 2020 dataset.
The system predicts binary machine failure, converts predictions into actionable risk/maintenance decisions,
and exposes everything through a professional Streamlit dashboard backed by a live IBM watsonx.ai Runtime online deployment.

**Scope:**
- Data validation → preprocessing → feature engineering → model training/comparison → IBM deployment
- Risk engine, priority engine, maintenance recommender, what-if simulator
- Streamlit dashboard connected to the deployed IBM model endpoint
- Full test suite + documentation + architecture diagram
- Security: zero hardcoded credentials; environment variables only

**Non-goals:**
- Real remaining-useful-life (RUL) time-series prediction
- Time-series data from a real production fleet
- Any non-IBM ML platform

**Key constraints:**
- Python 3.10+ ecosystem
- IBM watsonx.ai Runtime (ibm-watsonx-ai SDK) for model hosting
- IBM Cloud Object Storage automatically provisioned by watsonx.ai Studio project
- Logo file must be placed at `assets/yantraguard-logo.png` before the logo sub-task runs
- All IBM credentials supplied via environment variables (`.env` + `python-dotenv`; never committed)

---

## Architecture

```
AI4I 2020 Dataset (CSV)
        │
        ▼
IBM Cloud Object Storage (auto-provisioned bucket by watsonx.ai Studio project)
        │
        ▼
watsonx.ai Studio Jupyter Notebook
  ├── Data Validation Report
  ├── Preprocessing Pipeline (sklearn Pipeline object)
  ├── Feature Engineering
  ├── Leakage Investigation
  ├── Class Imbalance Analysis
  ├── Model Training: Logistic Regression │ Random Forest │ XGBoost
  ├── Fair Comparison (PR-AUC / ROC-AUC / F1/F2 / Calibration)
  ├── Threshold Optimisation
  ├── SHAP Explainability
  └── Model Serialisation (joblib pipeline)
        │
        ▼
watsonx.ai Runtime — Model Asset Store
        │
        ▼
Deployment Space → Online Deployment → Inference REST API
        │
        ▼
Streamlit Dashboard (local or IBM Code Engine)
  ├── Overview
  ├── Machine Health Prediction (→ calls IBM endpoint)
  ├── Risk Drivers / SHAP Explanation
  ├── Inspection Priority
  ├── Maintenance Recommendation
  ├── What-if Simulator (→ calls IBM endpoint)
  ├── Model Information
  └── Dataset / Validation Summary
```

---

## Folder Structure (Proposed)

```
YantraGuard/
├── assets/
│   └── yantraguard-logo.png          ← supplied logo (must be placed before implementation)
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv              ← untouched original dataset
│   └── processed/                    ← generated artefacts (gitignored)
├── notebooks/
│   └── 01_yantraguard_training.ipynb ← watsonx.ai Studio notebook
├── src/
│   ├── __init__.py
│   ├── config.py                     ← constants, thresholds, feature lists
│   ├── data_validation.py            ← data quality checks & report
│   ├── preprocessing.py              ← sklearn Pipeline factory
│   ├── feature_engineering.py        ← derived feature transformers
│   ├── train.py                      ← model training & comparison
│   ├── evaluate.py                   ← metrics, calibration, threshold search
│   ├── explainability.py             ← SHAP wrapper
│   ├── risk_engine.py                ← probability → risk level
│   ├── priority_engine.py            ← inspection priority rules
│   ├── recommender.py                ← maintenance recommendation rules
│   ├── ibm_client.py                 ← watsonx.ai Runtime API wrapper
│   └── utils.py                      ← shared helpers
├── dashboard/
│   ├── app.py                        ← main Streamlit entry point
│   └── pages/
│       ├── overview.py
│       ├── prediction.py
│       ├── risk_drivers.py
│       ├── priority.py
│       ├── recommendation.py
│       ├── whatif.py
│       ├── model_info.py
│       └── dataset_summary.py
├── tests/
│   ├── test_data_validation.py
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_risk_engine.py
│   ├── test_priority_engine.py
│   ├── test_recommender.py
│   ├── test_ibm_client.py
│   ├── test_pipeline_consistency.py
│   └── test_dashboard_inputs.py
├── docs/
│   ├── architecture.md
│   ├── data_dictionary.md
│   ├── model_methodology.md
│   ├── leakage_prevention.md
│   ├── evaluation_results.md          ← filled in after training
│   ├── deployment_instructions.md
│   ├── limitations.md
│   ├── future_improvements.md
│   └── user_instructions.md
├── .env.example                       ← template (no real values)
├── .gitignore
├── requirements.txt
├── README.md
└── PROJECT_REPORT.md
```

---

## Sub-Tasks

---

### Sub-Task 0 — Workspace Bootstrap & Logo Verification
**Status:** [ ] pending

**Intent:**
Establish the project folder skeleton, verify the logo asset is present,
and create the `.gitignore`, `.env.example`, and `requirements.txt`.

**Expected Outcomes:**
- All directories created
- `assets/yantraguard-logo.png` confirmed present (or user instructed to place it)
- `.gitignore` excludes `.env`, `data/processed/`, `__pycache__/`, model artefact files
- `.env.example` lists all required IBM credential keys (no real values)
- `requirements.txt` lists all Python dependencies

**Todo List:**
1. Check whether `assets/yantraguard-logo.png` exists; if not, halt and instruct user
2. Create directory tree: `assets/`, `data/raw/`, `data/processed/`, `src/`, `dashboard/pages/`, `tests/`, `docs/`, `notebooks/`
3. Write `.gitignore`
4. Write `.env.example` with keys: `IBM_CLOUD_API_KEY`, `IBM_WATSON_ML_INSTANCE_ID`, `IBM_WATSON_ML_URL`, `IBM_DEPLOYMENT_ID`, `IBM_SPACE_ID`
5. Write `requirements.txt`
6. Write `src/__init__.py`

**Relevant Context:**
- Logo currently exists at workspace root as `assetsyantraguardlogo.png.png` — user must place the correctly named file at `assets/yantraguard-logo.png`
- Python dependencies: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`, `plotly`, `ibm-watsonx-ai`, `python-dotenv`, `imbalanced-learn`, `pytest`, `joblib`

---

### Sub-Task 1 — Dataset Acquisition & Data Validation
**Status:** [ ] pending

**Intent:**
Load the AI4I 2020 dataset and produce a reproducible data-validation report
covering all checks required by Point 5. This report becomes the Dataset/Validation Summary
shown in the dashboard.

**Expected Outcomes:**
- `data/raw/ai4i2020.csv` present and untouched
- `src/data_validation.py` module with `validate_dataset(df) -> ValidationReport`
- A printed/JSON validation report with actual numbers (row count, column names/types, missing values, duplicates, class distribution, numeric stats)
- `tests/test_data_validation.py` passing

**Todo List:**
1. Confirm AI4I 2020 CSV column names exactly (UDI, Product ID, Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min], Machine failure, TWF, HDF, PWF, OSF, RNF)
2. Write `src/data_validation.py`: row/column count, dtypes, missing values, duplicate check, target distribution, value range checks per feature, statistical summary
3. Write `ValidationReport` dataclass to carry structured results
4. Write `tests/test_data_validation.py`

**Relevant Context:**
- Dataset source: UCI ML Repository — AI4I 2020 Predictive Maintenance Dataset (10,000 rows × 14 columns)
- Target: `Machine failure` (binary 0/1; expected ~3.4% failure rate — class imbalance is significant)

---

### Sub-Task 2 — Leakage Prevention & Feature Selection
**Status:** [ ] pending

**Intent:**
Explicitly document and enforce leakage prevention.
Define the exact model input feature list.
This is the single source of truth for all downstream components.

**Expected Outcomes:**
- `src/config.py` defines `EXCLUDED_FEATURES`, `INPUT_FEATURES`, `TARGET`, `FEATURE_DESCRIPTIONS`
- Leakage rationale documented in `docs/leakage_prevention.md`

**Todo List:**
1. Write `src/config.py` with:
   - `TARGET = "Machine failure"`
   - `EXCLUDED_FEATURES = ["UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]`
   - `INPUT_FEATURES = ["Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]`
   - Risk thresholds: `LOW < 0.20`, `MODERATE 0.20–0.50`, `HIGH 0.50–0.75`, `CRITICAL >= 0.75`
   - Feature bounds for input validation
2. Write `docs/leakage_prevention.md` explaining each excluded variable and why

**Relevant Context:**
- `TWF`, `HDF`, `PWF`, `OSF`, `RNF` are sub-failure mode indicators that are components of the `Machine failure` target — using them would be direct target leakage
- `UDI` and `Product ID` are identifiers with no predictive physical meaning
- `Machine failure` is the target itself

---

### Sub-Task 3 — Preprocessing Pipeline & Feature Engineering
**Status:** [ ] pending

**Intent:**
Build a reproducible sklearn `Pipeline` that encodes `Type`, scales numeric features,
and optionally adds physically meaningful engineered features.
The pipeline must prevent leakage during cross-validation and must be saved as a reusable artefact.

**Expected Outcomes:**
- `src/preprocessing.py` with `build_preprocessing_pipeline()` factory
- `src/feature_engineering.py` with a custom `sklearn` transformer adding: `temp_diff`, `mechanical_power`, `torque_x_tool_wear`
- `tests/test_preprocessing.py` and `tests/test_feature_engineering.py` passing
- Pipeline can be serialised/deserialised with `joblib`

**Todo List:**
1. Write `src/feature_engineering.py` with `FeatureEngineer(BaseEstimator, TransformerMixin)` that adds derived columns:
   - `temp_diff = Process temperature [K] - Air temperature [K]`
   - `mechanical_power = Torque [Nm] × Rotational speed [rpm] × (2π/60)`
   - `torque_x_tool_wear = Torque [Nm] × Tool wear [min]`
2. Write `src/preprocessing.py` with pipeline: OrdinalEncoder for `Type` → FeatureEngineer → StandardScaler
3. Ensure FeatureEngineer is inside the pipeline so it is properly fitted on training data only
4. Write `tests/test_preprocessing.py` and `tests/test_feature_engineering.py`

**Relevant Context:**
- `Type` is categorical (L, M, H) — use OrdinalEncoder with known categories
- StandardScaler must be fit only on training split to avoid leakage
- Feature engineering evaluation (with vs without) is done in Sub-Task 4

---

### Sub-Task 4 — Model Training, Comparison & Selection
**Status:** [ ] pending

**Intent:**
Train Logistic Regression, Random Forest, and XGBoost.
Perform stratified train/val/test split.
Compare models using PR-AUC, ROC-AUC, F1, F2, Recall, Precision, Calibration.
Evaluate feature engineering contribution (with/without).
Address class imbalance with class_weight and threshold tuning.
Select the best model objectively.

**Expected Outcomes:**
- `src/train.py` with `train_models(X_train, y_train) -> dict[str, Pipeline]`
- `src/evaluate.py` with `evaluate_model(model, X_test, y_test, threshold) -> EvaluationResult`
- Actual metric table comparing all three models on the held-out test set
- Threshold optimisation results documented
- Selected model + threshold saved to `data/processed/best_model.joblib`
- `docs/evaluation_results.md` filled with actual numbers post-training
- `docs/model_methodology.md`

**Todo List:**
1. Write `src/train.py`:
   - Stratified 70/15/15 train/val/test split with `random_state=42`
   - Apply SMOTE only within training fold (use `imblearn.pipeline.Pipeline`)
   - Logistic Regression with `class_weight="balanced"`
   - Random Forest with `class_weight="balanced"`, reasonable hyperparameters
   - XGBoost with `scale_pos_weight` set to negative/positive class ratio
   - Run with and without feature engineering and record metric differences
2. Write `src/evaluate.py`:
   - `compute_metrics()`: precision, recall, F1, F2, ROC-AUC, PR-AUC, confusion matrix
   - `find_optimal_threshold()`: maximise F2 on validation set (prioritises recall for safety)
   - `calibrate_if_needed()`: Platt scaling / isotonic regression if calibration curve is poor
3. Persist best model + preprocessor as single `joblib` Pipeline to `data/processed/best_model.joblib`
4. Write `docs/model_methodology.md` and `docs/evaluation_results.md` (placeholders; filled with real numbers after execution)

**Relevant Context:**
- F2 score (β=2) weights recall twice as much as precision — appropriate for failure detection where missing a failure is more costly than a false alarm
- Calibration check: use `sklearn.calibration.calibration_curve` on val set; apply `CalibratedClassifierCV` if needed
- SMOTE must be inside pipeline or applied only to training fold data — never before the split

---

### Sub-Task 5 — Risk Engine, Priority Engine & Maintenance Recommender
**Status:** [ ] pending

**Intent:**
Convert raw failure probability into human-readable risk level, inspection priority score,
and a practical maintenance recommendation. All three components are rule-based, clearly labeled as such.

**Expected Outcomes:**
- `src/risk_engine.py` with `assess_risk(probability) -> RiskAssessment`
- `src/priority_engine.py` with `compute_priority(risk, feature_values) -> PriorityResult`
- `src/recommender.py` with `generate_recommendation(risk, feature_values, top_drivers) -> Recommendation`
- Tests passing for all three modules

**Todo List:**
1. Write `src/risk_engine.py`:
   - `RiskLevel` enum: LOW / MODERATE / HIGH / CRITICAL
   - Thresholds from `config.py` (configurable)
   - Returns: `RiskAssessment(probability, risk_level, confidence_band)`
2. Write `src/priority_engine.py`:
   - Priority score = weighted combination of risk_level ordinal + tool_wear normalised + torque normalised
   - Returns: `PriorityResult(score, label, rationale)` where label is "Routine" / "Monitor" / "Schedule" / "Immediate"
   - Docstring explicitly states this is rule-based, not a learned ranking
3. Write `src/recommender.py`:
   - Rule tree: if CRITICAL → immediate shutdown/inspection; if HIGH → schedule within 24h; if MODERATE → inspect at next planned stop; if LOW → continue monitoring
   - Augment with feature-specific notes (e.g. high tool wear → "Check/replace cutting tool")
   - Returns: `Recommendation(action, rationale, caveats)`
4. Write tests for all three

**Relevant Context:**
- Risk thresholds defined in `src/config.py` (Sub-Task 2)
- Observation-level priority only — the dataset has no longitudinal machine history; do not claim fleet-level ranking

---

### Sub-Task 6 — SHAP Explainability
**Status:** [ ] pending

**Intent:**
Compute SHAP values for individual predictions to identify top risk drivers.
Provide a model-agnostic explanation that can be displayed in the dashboard.

**Expected Outcomes:**
- `src/explainability.py` with `explain_prediction(model_pipeline, input_df, background_data) -> ExplanationResult`
- Returns top N feature contributions with direction (positive/negative impact)
- Clear disclaimer: SHAP values explain the model's decision, not physical root cause

**Todo List:**
1. Write `src/explainability.py`:
   - Use `shap.TreeExplainer` for Random Forest / XGBoost (fast, exact)
   - Use `shap.LinearExplainer` for Logistic Regression
   - Extract top-5 contributors with signed SHAP values
   - Cache explainer object to avoid re-initialising on every call
2. Return `ExplanationResult(feature_names, shap_values, base_value, disclaimer)`
3. Handle pipeline transform: pass data through preprocessing steps before SHAP

**Relevant Context:**
- SHAP must operate on the transformed feature space (after OrdinalEncoder + FeatureEngineer + Scaler)
- The background dataset for TreeExplainer should be a sample of training data (100–200 rows), kept in memory

---

### Sub-Task 7 — IBM watsonx.ai Runtime Client
**Status:** [ ] pending

**Intent:**
Build the IBM Cloud integration layer.
Wrap the `ibm-watsonx-ai` Python client to:
- Authenticate using environment-variable credentials (IAM API key)
- Store the trained pipeline as a model asset in the watsonx.ai Runtime repository
- Create/reference a deployment space
- Create an online deployment
- Score new observations via the REST inference endpoint

**Expected Outcomes:**
- `src/ibm_client.py` with `WatsonxClient` class
- `deploy_model(model_path)` function that uploads and deploys the best model
- `predict(features_dict) -> PredictionResult` function that calls the live endpoint
- `tests/test_ibm_client.py` with mocked IBM API calls (no real credentials needed for unit tests)
- `docs/deployment_instructions.md`

**Todo List:**
1. Write `src/ibm_client.py`:
   - Load credentials from env vars: `IBM_CLOUD_API_KEY`, `IBM_WATSON_ML_URL`, `IBM_WATSON_ML_INSTANCE_ID`, `IBM_DEPLOYMENT_ID`, `IBM_SPACE_ID`
   - `WatsonxClient.__init__`: instantiate `ibm_watsonx_ai.APIClient` with `IAMTokenManager`
   - `deploy_model(model_path, model_name, software_spec)`: store model → create deployment → return deployment_id
   - `score(input_records: list[dict]) -> list[float]`: call `client.deployments.score()` with correct payload format
   - Graceful fallback: if `IBM_DEPLOYMENT_ID` not set, load local `best_model.joblib` and warn
2. Write `tests/test_ibm_client.py` using `unittest.mock.patch`
3. Write `docs/deployment_instructions.md` with step-by-step IBM Cloud setup guide

**Relevant Context:**
- IBM watsonx.ai Python SDK: `ibm-watsonx-ai` (successor to `ibm-watson-machine-learning`)
- Authentication: `from ibm_watsonx_ai import APIClient, Credentials`
- Scoring payload format: `{"input_data": [{"fields": [...], "values": [[...]]}]}`
- Software spec for scikit-learn pipelines: `"runtime-23.1-py3.10"` (verify current spec name in IBM docs or deployment notebook)
- Deployment space must be created manually in watsonx.ai Studio UI first; `IBM_SPACE_ID` is then copied to `.env`

---

### Sub-Task 8 — Streamlit Dashboard
**Status:** [ ] pending

**Intent:**
Build the professional Streamlit dashboard with all 8 required sections.
Connect prediction flow to the IBM watsonx.ai Runtime endpoint.
Apply YantraGuard AI branding (dark industrial theme, blue/cyan accents).

**Expected Outcomes:**
- `dashboard/app.py` entry point with sidebar navigation and logo
- 8 page modules in `dashboard/pages/` all functional
- Input validation prevents invalid values from reaching the model
- Prediction flow: user inputs → ibm_client.score() → risk_engine → priority_engine → recommender → display
- What-if simulator calls the same prediction flow with modified values
- `tests/test_dashboard_inputs.py` validates input bounds

**Todo List:**
1. Write `dashboard/app.py`:
   - Load and display `assets/yantraguard-logo.png` in sidebar header
   - Dark theme CSS injection (background `#0d1117`, accent `#00b4d8`)
   - Sidebar navigation linking all 8 pages
2. Write `dashboard/pages/overview.py`: project introduction, architecture diagram text, tagline
3. Write `dashboard/pages/prediction.py`:
   - Sliders/selectors for: Type (L/M/H), Air Temp (K), Process Temp (K), RPM, Torque (Nm), Tool Wear (min)
   - Input validation against bounds from `config.py`
   - On "Predict" button: call `ibm_client.score()` → display probability gauge, risk badge, priority label
4. Write `dashboard/pages/risk_drivers.py`: SHAP bar chart with signed contributions, disclaimer
5. Write `dashboard/pages/priority.py`: priority score card, rationale, observation-level caveat
6. Write `dashboard/pages/recommendation.py`: formatted recommendation with action, rationale, caveats
7. Write `dashboard/pages/whatif.py`: side-by-side baseline vs modified scenario comparison
8. Write `dashboard/pages/model_info.py`: model name, threshold, training metrics table, calibration plot
9. Write `dashboard/pages/dataset_summary.py`: validation report rendered as tables/charts
10. Write `tests/test_dashboard_inputs.py`

**Relevant Context:**
- Streamlit `st.set_page_config(layout="wide", page_icon="assets/yantraguard-logo.png")`
- Use `plotly` for gauges and SHAP bar charts (renders in Streamlit without extra config)
- Session state to cache model predictions and avoid re-calling IBM endpoint on every rerender

---

### Sub-Task 9 — Training Notebook (watsonx.ai Studio)
**Status:** [ ] pending

**Intent:**
Create a self-contained Jupyter notebook that executes the full ML pipeline
inside watsonx.ai Studio, uploads the dataset to Cloud Object Storage,
trains models, evaluates them, and deploys the best model to watsonx.ai Runtime.

**Expected Outcomes:**
- `notebooks/01_yantraguard_training.ipynb` ready to run inside watsonx.ai Studio
- Notebook cells cover: COS data load → validation → preprocessing → training → evaluation → SHAP → model persist → deploy
- All IBM credential cells use environment variable reads, not hardcoded values
- Deployment cell outputs actual deployment ID to be copied into `.env`

**Todo List:**
1. Create notebook sections:
   - Cell 0: Install dependencies (`!pip install ibm-watsonx-ai shap imbalanced-learn xgboost`)
   - Cell 1: Imports and credential loading from env / project token
   - Cell 2: Load dataset from COS project bucket
   - Cell 3: Data validation (call `src/data_validation.py` logic inline or import)
   - Cell 4: Preprocessing + feature engineering
   - Cell 5: Train/val/test split
   - Cell 6: Train all 3 models, evaluate, compare
   - Cell 7: Threshold optimisation + calibration
   - Cell 8: SHAP global importance plots
   - Cell 9: Save best model with `joblib`
   - Cell 10: Persist model to watsonx.ai Runtime repository
   - Cell 11: Create online deployment, print deployment_id and scoring URL
2. Add markdown cells explaining each decision

**Relevant Context:**
- Inside watsonx.ai Studio, project token gives automatic COS access via `project.get_file()`
- Software spec to use: `"runtime-23.1-py3.10"` or latest available — must be verified in Studio UI
- Model type for sklearn Pipeline: `"scikit-learn_1.1"` or appropriate version

---

### Sub-Task 10 — Test Suite
**Status:** [ ] pending

**Intent:**
Ensure all major modules have test coverage and can be run with `pytest`.
Tests must not require live IBM credentials.

**Expected Outcomes:**
- All test files in `tests/` pass with `pytest` from project root
- IBM client tests use mocks
- No test imports `.env` secrets

**Todo List:**
1. Verify all test files written in previous sub-tasks exist
2. Write any missing test stubs
3. Add `conftest.py` with shared fixtures (sample DataFrame, sample features dict)
4. Run `pytest tests/ -v` and confirm all pass
5. Document test results in `README.md`

**Relevant Context:**
- Use `unittest.mock.patch` for `ibm_watsonx_ai.APIClient`
- Use synthetic data matching AI4I schema for validation/preprocessing tests

---

### Sub-Task 11 — Documentation & README
**Status:** [ ] pending

**Intent:**
Complete all documentation files and the main README with actual results filled in after training.

**Expected Outcomes:**
- `README.md` complete with: project description, architecture, setup instructions, run instructions, IBM Cloud setup, limitations
- All `docs/` files complete
- `PROJECT_REPORT.md` complete

**Todo List:**
1. Write `README.md`
2. Write `docs/architecture.md` (text version of architecture diagram)
3. Write `docs/data_dictionary.md` (all 14 AI4I columns described)
4. Write `docs/model_methodology.md` (training strategy, imbalance handling, threshold rationale)
5. Write `docs/leakage_prevention.md` (already started in Sub-Task 2)
6. Fill `docs/evaluation_results.md` with actual metrics after Sub-Task 4 runs
7. Write `docs/deployment_instructions.md` (already started in Sub-Task 7)
8. Write `docs/limitations.md`
9. Write `docs/future_improvements.md`
10. Write `docs/user_instructions.md`
11. Write `PROJECT_REPORT.md`

---

### Sub-Task 12 — Final Validation & Demonstration Checklist
**Status:** [ ] pending

**Intent:**
Run the full end-to-end check before delivery.
Verify every requirement from the 35-point list is met.

**Expected Outcomes:**
- All 35 requirement points verified
- No hardcoded credentials in any file
- `pytest` passes
- Dashboard launches with `streamlit run dashboard/app.py`
- IBM deployment ID confirmed working (or limitation stated clearly)
- Final demonstration checklist produced

**Todo List:**
1. Run `grep -r "API_KEY\|password\|secret" src/ dashboard/` to confirm no hardcoded secrets
2. Run `pytest tests/ -v` — confirm all pass
3. Run `streamlit run dashboard/app.py` — confirm dashboard loads
4. Verify logo is displayed in dashboard header
5. Verify prediction flow reaches IBM endpoint (or documents fallback mode)
6. Run through all 35 requirement points and check each
7. Write final demonstration checklist as `DEMO_CHECKLIST.md`

---

## Implementation Notes & Assumptions

| # | Note |
|---|------|
| A1 | AI4I 2020 CSV must be downloaded by the user from UCI ML Repository and placed at `data/raw/ai4i2020.csv`. The dataset is publicly available but cannot be auto-downloaded in all environments. |
| A2 | IBM watsonx.ai Studio deployment space must be created manually in the IBM Cloud UI. The resulting `SPACE_ID` is configured in `.env`. |
| A3 | The `ibm-watsonx-ai` SDK software spec name (`"runtime-23.1-py3.10"`) should be verified in the Studio UI at deployment time; spec names change with SDK versions. |
| A4 | SHAP computation is done locally (in the dashboard process) using the locally-loaded pipeline for explainability. The IBM endpoint is used for the primary probability prediction. |
| A5 | The AI4I 2020 dataset is a synthetic benchmark, not a real industrial fleet. All outputs are clearly labeled as model-based decision support. |
| A6 | Logo file `assetsyantraguardlogo.png.png` exists at workspace root but must be renamed/moved to `assets/yantraguard-logo.png` by the user before Sub-Task 0 completes. |

---

## Open Questions for User Confirmation

1. **IBM Cloud region** — Which region will you use for watsonx.ai (e.g., `us-south` Dallas, `eu-gb` London)? This affects endpoint URLs and available software specs.
2. **Streamlit deployment** — Should the Streamlit app run locally only, or should it also be deployed (e.g., IBM Cloud Code Engine / Streamlit Community Cloud)?
3. **Dataset download** — Do you already have `ai4i2020.csv` available, or should the notebook include download instructions from UCI ML Repository?
4. **Feature engineering toggle** — Should the dashboard allow users to toggle engineered features on/off, or always use whichever configuration produced the best model?
5. **SHAP in dashboard** — SHAP computation for tree models can be slow on first load. Is a slight delay (2–5 seconds for background data init) acceptable, or should we cache SHAP explanations per session?
