<div align="center">

<img src="assets/logo1.png" alt="Octal Logo" width="220"/>

# OCTAL Cars Price Evaluation System

**The AI/backend engine behind Octal — estimating vehicle market value to support insurance underwriting, policy pricing, and claims settlement in the Egyptian market**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-Validation-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Supabase](https://img.shields.io/badge/Database-Supabase-3FCF8E?logo=supabase&logoColor=white)](https://supabase.com/)

</div>

---

An end-to-end machine learning project that predicts used vehicle prices from listing attributes (make, model, year, mileage, engine capacity, trim, body type, etc.). The project covers the full pipeline: data cleaning, EDA, feature engineering, model training and tuning, a production FastAPI backend (including a VIN-decoding prediction flow), and deployment to FastAPI Cloud with Supabase as a persistent operation log.

**Live API:** deployed on [FastAPI Cloud](https://fastapicloud.dev)
**Repository:** [github.com/Eyad-Elghonemy/supbase-API](https://github.com/Eyad-Elghonemy/supbase-API)


> ⚠️ **Security note:** the `X-API-Key` below is a **demo key** shared for evaluation purposes only. If this repository is public, rotate the key (set a new `SECRET_KEY_TOKEN` in your Hugging Face Space secrets) so this value stops working. Never rely on a key that has appeared in a public README for anything beyond a quick demo.

```
Demo X-API-Key: c0c2d9d05029aed5d5174ff5ff8e6d88
```

---


> This repository contains the **model training notebook and backend API only**. The frontend that consumes this API is maintained in a separate repository.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Modeling Journey](#-modeling-journey)
- [Results](#-results)
- [Diagnostic Plots](#-diagnostic-plots)
- [Feature Importance](#-feature-importance)
- [Which Model Is Actually Deployed](#-which-model-is-actually-deployed)
- [Backend Architecture](#-backend-architecture)
- [The VIN-Based Prediction Feature](#-the-vin-based-prediction-feature)
- [Operation Logging & Supabase](#-operation-logging--supabase)
- [API Reference](#-api-reference)
- [Deployment (FastAPI Cloud)](#-deployment-fastapi-cloud)
- [Deployment Issues Encountered & Fixed](#-deployment-issues-encountered--fixed)
- [Project Structure](#-project-structure)
- [Environment Variables](#-environment-variables)
- [Installation (Local)](#-installation-local)
- [Notes & Limitations](#-notes--limitations)
- [Future Work](#-future-work)

---

## 🔎 Overview

The goal of this project is to build a model that estimates a used vehicle's price based on its specifications, and to serve that model through a real, deployed API — not just a notebook. The pipeline covers:

1. Data cleaning and outlier handling
2. Exploratory Data Analysis (univariate, bivariate, multivariate)
3. Feature selection using correlation (Pearson / Spearman / Kendall) and ANOVA (p-value) tests
4. Feature engineering (estimating `engine_capacity` from horsepower)
5. Preprocessing (scaling, ordinal/label/one-hot encoding)
6. Model training, evaluation, and hyperparameter tuning
7. Feature importance analysis
8. A FastAPI backend (this repo) serving predictions from raw vehicle data **and** from a VIN alone — designed to be consumed by any external client (a separately maintained frontend, mobile app, or other service)
9. A protected operation-log endpoint backed by Supabase (a managed external database)
10. Deployment to FastAPI Cloud

**Target variable:** `price` (modeled as `log1p(price)` to reduce right-skew; predictions are converted back with `expm1`)

---

## 🗂 Dataset

- **Raw size:** 1,000,000 rows × 21 columns
- **After cleaning (removing price/mileage outliers):** ~945,955 rows × 12 columns
- **Train / Test split:** 70% / 30% (`random_state=45`)
  - `X_train`: 662,168 rows
  - `X_test`: 283,787 rows

### Features kept for modeling
| Type | Columns |
|---|---|
| Numerical | `year`, `mileage`, `engine_capacity` |
| Ordinal | `trim` (Base → LX → EX → Sport → Touring → Limited) |
| High-cardinality categorical | `make`, `model` |
| Nominal categorical | `transmission`, `fuel_type`, `drivetrain`, `body_type` |

### Columns dropped
Based on domain reasoning and weak statistical relevance to `price`:
`vehicle_age`, `engine_hp`, `owner_count`, `mileage_per_year`, `brand_popularity`, `interior_color`, `condition`, `seller_type`, `exterior_color`, `accident_history`

---

## 🔬 Modeling Journey

1. **Cleaning:** stripped column names, estimated `engine_capacity` from `engine_hp`
2. **Feature relevance check:** Pearson / Spearman / Kendall correlation for numerical features vs. `price`; one-way ANOVA (p-value) for categorical features vs. `price`
3. **Outlier handling:** removed a large spike of listings capped at `price == 1500`; removed listings with `mileage >= 300,000`
4. **Transformation:** applied `log1p` on `price` to normalize its distribution
5. **EDA:** distribution plots, price-by-make/transmission/fuel/body-type/trim boxplots, price vs. mileage/engine capacity scatter plots

### Preprocessing pipeline

Built with `ColumnTransformer` + `Pipeline`:

| Column group | Columns | Transformer |
|---|---|---|
| Numerical | `year`, `mileage`, `engine_capacity` | `StandardScaler` |
| Ordinal | `trim` | `OrdinalEncoder` (custom order) |
| Label | `make`, `model` | `OrdinalEncoder` (unknown → -1) |
| Nominal | `transmission`, `fuel_type`, `drivetrain`, `body_type` | `OneHotEncoder(drop='first')` |

Final feature matrix shape: **17 columns** after encoding. The fitted preprocessor is saved to `models/preprocessor.pkl` and reused identically at inference time by the API.

### Models trained

**1. Linear Regression (baseline)**
A simple baseline to confirm whether the target requires a non-linear model. Residual analysis showed a curved (non-random) pattern — clear evidence of underfitting relative to the tree-based alternatives.

**2. Random Forest Regression — Base**
```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=45,
    n_jobs=-1
)
```

**3. Random Forest Regression — Tuned (RandomizedSearchCV)**
A `RandomizedSearchCV` (`cv=3`, `n_iter=15`, scoring=`neg_root_mean_squared_error`) was run once on Google Colab over:
```python
params_narrow = {
    'n_estimators': [150, 200, 250],
    'max_depth': [15, 20, 25],
    'min_samples_split': [3, 5, 8],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt']
}
```
**Best parameters found:**
```python
RandomForestRegressor(
    n_estimators=250, max_depth=25, min_samples_split=3,
    min_samples_leaf=1, max_features='sqrt', random_state=45, n_jobs=-1
)
```

> ⚠️ The tuned model (~7 GB on disk) was trained and evaluated on Colab but could not be downloaded/committed due to its size. Only the best hyperparameters and the resulting metrics were kept — the model file itself is **not included** in this repo.

---

## 📊 Results

Evaluated on `price_log` (RMSE/MAE are in log-price units):

| Model | Split | RMSE ↓ | MAE ↓ | R² ↑ | Usage |
|---|---|---|---|---|---|
| Linear Regression | Train | 0.2040 | 0.1504 | 0.9162 | ✅ **Used for hosting/deployment** (small, fast to load) |
| Linear Regression | Test | 0.2040 | 0.1502 | 0.9163 | ✅ **Used for hosting/deployment** (small, fast to load) |
| Random Forest (base) | Train | 0.0886 | 0.0669 | 0.9840 | 🖥️ Used locally only (not deployed) |
| Random Forest (base) | Test | 0.1160 | 0.0872 | 0.9726 | 🖥️ Used locally only (not deployed) |
| **Random Forest (tuned)** | **Train** | **0.0565** | **0.0430** | **0.9935** | ❌ Not used at all |
| **Random Forest (tuned)** | **Test** | **0.1134** | **0.0859** | **0.9738** | ❌ Not used at all |

**Why the tuned Random Forest was dropped entirely:** its file size (~7 GB) made it impractical to store/deploy, its test-set performance was only marginally better than the base model (RMSE 0.1134 vs. 0.1160 — a ~2% improvement), and the widening gap between its train and test scores is an early sign of overfitting. The base Random Forest was judged the better trade-off of the two tree models, but even it wasn't carried into the hosted API — see [Which Model Is Actually Deployed](#-which-model-is-actually-deployed).

**Takeaways:**
- Linear Regression fails to capture the non-linear relationship between features and price (R² ≈ 0), confirming the need for a tree-based model.
- Random Forest dramatically improves performance, explaining ~97–99% of price variance.
- Tuning improves training fit further but yields only a marginal test-set gain over the base model, and the train/test gap widens slightly — suggesting the tuned model sits closer to (but not clearly past) the overfitting boundary. The base model remains a strong, more conservative choice.

---

## 🌟 Feature Importance

Computed from the **base Random Forest** model (`forest.feature_importances_`) after preprocessing:

| Rank | Feature | Importance |
|---|---|---|
| 1 | `engine_capacity` | 0.3318 |
| 2 | `year` | 0.3188 |
| 3 | `mileage` | 0.2250 |
| 4 | `make` | 0.0774 |
| 5 | `model` | 0.0279 |
| 6 | `body_type_Pickup Truck` | 0.0049 |
| 7 | `body_type_Hatchback` | 0.0035 |
| 8 | `trim` | 0.0023 |
| 9 | `body_type_SUV` | 0.0015 |
| 10 | `body_type_Sedan` | 0.0013 |
| 11 | `fuel_type_Electric` | 0.0011 |
| 12 | `body_type_Minivan` | 0.0011 |
| 13 | `body_type_Wagon` | 0.0009 |
| 14 | `transmission_Manual` | 0.0008 |
| 15 | `drivetrain_FWD` | 0.0007 |
| 16 | `drivetrain_RWD` | 0.0007 |
| 17 | `fuel_type_Gasoline` | 0.0007 |

**Interpretation:** `engine_capacity`, `year`, and `mileage` together drive over **87%** of the model's predictive power, while `make` adds meaningful brand-level signal. Categorical dummy variables (body type, fuel type, drivetrain, transmission) contribute comparatively little individually — this is exactly why the VIN prediction flow (below) is comfortable falling back to reasonable estimates for those fields when NHTSA doesn't provide them, while treating `make`, `model`, and `year` as non-negotiable.

---

## 📉 Diagnostic Plots

For each model, three diagnostic plots were generated: **Actual vs. Predicted** (points should hug the red diagonal), a **Residual Plot** (residuals should scatter randomly around zero with no shape), and a **Distribution of Residuals** (should be roughly symmetric and centered on zero).

> Place the corresponding image files under `assets/` using the paths referenced below.

### 1. Linear Regression

| Actual vs. Predicted | Residual Plot | Residual Distribution |
|---|---|---|
| ![LR Actual vs Predicted](assets/lr_actual_vs_predicted.png) | ![LR Residual Plot](assets/lr_residual_plot.png) | ![LR Residual Distribution](assets/lr_residual_distribution.png) |

Predictions follow the diagonal reasonably well in the mid-range but drift away at both extremes, overpredicting cheap vehicles and underpredicting expensive ones. The residual plot shows a clear curved (non-random) shape rather than random scatter, and the residual distribution is visibly skewed with a long negative tail — all consistent evidence that a linear model is underfitting a genuinely non-linear relationship, and the motivation for moving to tree-based models.

### 2. Random Forest (base)

| Actual vs. Predicted | Residual Plot | Residual Distribution |
|---|---|---|
| ![RF Base Actual vs Predicted](assets/rf_base_actual_vs_predicted.png) | ![RF Base Residual Plot](assets/rf_base_residual_plot.png) | ![RF Base Residual Distribution](assets/rf_base_residual_distribution.png) |

Points track the diagonal much more tightly than Linear Regression across the full price range. Residuals are centered on zero with a fairly symmetric, narrow spread and no strong systematic curvature, and the residual distribution is sharply peaked around zero — a substantially better fit than the linear baseline, consistent with the large jump in R² reported above.

### 3. Random Forest (tuned)

| Actual vs. Predicted | Residual Plot | Residual Distribution |
|---|---|---|
| ![RF Tuned Actual vs Predicted](assets/rf_tuned_actual_vs_predicted.png) | ![RF Tuned Residual Plot](assets/rf_tuned_residual_plot.png) | ![RF Tuned Residual Distribution](assets/rf_tuned_residual_distribution.png) |

Visually very similar to the base Random Forest — the diagonal fit and residual spread are comparable, with a marginally tighter residual distribution. This visual similarity mirrors the [Results](#-results) table: tuning improved the test-set metrics only slightly over the base model, which is part of why the base model — not the tuned one — was judged the better trade-off (see [Why the tuned Random Forest was dropped entirely](#-results)).

---

## 🚀 Which Model Is Actually Deployed

This is an important, honest distinction:

| Model | Trained? | Evaluated? | Deployed to the API? |
|---|---|---|---|
| Linear Regression | ✅ | ✅ | ✅ **Yes — currently powers `/predict/linear` and `/predict/from-vin`** |
| Random Forest (base) | ✅ | ✅ | 🖥️ Used locally only during development; not wired into the hosted API |
| Random Forest (tuned) | ✅ | ✅ | ❌ Never — could not be exported (~7 GB) and showed early signs of overfitting |

**Why Linear Regression, despite Random Forest being clearly more accurate?** The base Random Forest artifact was not carried into the final backend configuration used for this deployment; only `preprocessor.pkl` and `lin_base.pkl` are loaded by `utils/config.py`. This keeps the deployed model small, fast to load, and easy to host on a free-tier container — at the cost of the accuracy gap documented above. Swapping in `forest_base.pkl` is a drop-in change (see [Future Work](#-future-work)).

---

## 🧩 Backend Architecture

The API is built with **FastAPI**, with all business logic separated out of `main.py` into a `utils` package:

| Module | Responsibility |
|---|---|
| `main.py` | Route definitions only |
| `utils/config.py` | Loads env vars and model artifacts once at startup |
| `utils/CustomerData.py` | Pydantic schema for a full, direct prediction request; also the final validation layer for VIN-based predictions |
| `utils/inference.py` | Runs the preprocessor + model, converts `price_log` back to `price` |
| `utils/VinRequestData.py` | Pydantic schema for the VIN-based request body |
| `utils/vin_decoder.py` | Calls the NHTSA API and translates its raw output into this project's feature values |
| `utils/vin_mapping.json` | Lookup table translating NHTSA's raw category strings into this model's trained categories |
| `utils/vin_prediction.py` | Orchestrates the full VIN → prediction flow, including fallback estimation |
| `utils/error_messages.py` | Translates every Pydantic validation error into a clear Arabic message |
| `utils/usage_tracker.py` | Operation logging, backed by Supabase |

### Request validation

`CustomerData` (used directly by `/predict/linear`, and as the final gate for `/predict/from-vin`) enforces:
- `make` / `model` / `body_type` / `fuel_type` / `drivetrain` / `trim` / `transmission` restricted to `Literal` sets matching the model's trained categories
- `year`: 1990–2025
- `mileage`: 500–300,000 miles
- `engine_capacity`: 0.0–4.0 liters, with a custom `model_validator` enforcing that electric vehicles report exactly `0` and every other fuel type reports at least `0.5`

### Arabic error messages (with correct RTL rendering)

Every validation failure — missing fields, invalid enum values, out-of-range numbers, custom validator errors — is translated into a ready-to-display Arabic sentence via `utils/error_messages.py`, rather than FastAPI's default technical English/JSON error format.

A specific bidi (bidirectional text) issue was identified and fixed: Arabic (RTL) sentences containing embedded Latin content (numbers, English category names) rendered with a scrambled word order in some clients, because the Unicode bidirectional algorithm doesn't always isolate LTR runs correctly on its own. The fix wraps any embedded LTR content with Unicode directional isolate marks (`U+2066` … `U+2069`), forcing correct rendering regardless of the client displaying the JSON.

---

## 🔑 The VIN-Based Prediction Feature

Beyond accepting full vehicle data directly, the API can also predict a price from **just a VIN plus three fields the VIN can't reliably provide** (`mileage`, `trim`, `transmission`).

### How it works

1. **Decode:** the VIN is sent to the free, public [NHTSA vPIC API](https://vpic.nhtsa.dot.gov/api/) (`DecodeVinValues`), which returns raw manufacturer data (make, model, model year, body class, fuel type, drive type, engine displacement, etc.).
2. **Translate:** raw NHTSA category strings (e.g. `"Sport Utility Vehicle (SUV)"`) are mapped to this model's trained categories (e.g. `"SUV"`) via `vin_mapping.json`.
3. **Round `engine_capacity` to the nearest 0.5L:** the training data only ever contains multiples of 0.5, so any decoded displacement is snapped to the nearest 0.5 (e.g. `2.3` → `2.5`) instead of being passed through as a continuous value the model never saw during training.
4. **Hard requirements — never guessed:** `make`, `model`, and `year` must resolve to values the model was actually trained on. If any of the three can't be reliably determined, the prediction is skipped entirely and the response reports back exactly which field(s) couldn't be resolved (`missing_make` / `missing_model` / `missing_year`), along with whatever was successfully extracted.
5. **Soft fields — estimated, not guessed silently:** `body_type`, `fuel_type`, `drivetrain`, and `engine_capacity` fall back to simple, explainable estimates (e.g. body-type-conditioned defaults) whenever NHTSA doesn't provide a usable value. Every estimated field is flagged explicitly in the response (`estimated_body_type`, `estimated_fuel_type`, `estimated_drivetrain`, `estimated_engine_capacity`), so the frontend can be transparent with the user about which numbers were decoded versus estimated.
6. **Merge & validate:** the resolved VIN data is merged with the user-supplied `mileage`, `trim`, and `transmission`, then re-validated end-to-end through `CustomerData` — the same Pydantic model used for direct predictions — before ever reaching the trained model.
7. **Predict:** the merged, validated payload is run through the same preprocessing pipeline and model as `/predict/linear`.

### Design rationale
This is a deliberate, explicit policy: **never silently fabricate the fields that matter most** (make/model/year drive the largest share of predictive signal — see [Feature Importance](#-feature-importance)), but **do provide a best-effort prediction** using low-importance fields even when they can't be fully confirmed, as long as that estimation is clearly flagged rather than hidden.

---

## 🗄 Operation Logging & Supabase

Every call to `/predict/linear` and `/predict/from-vin` (successful or not) is logged as:
```json
{"timestamp": "...", "operation_type": "linear" | "from-vin", "success": true | false}
```

### Why Supabase instead of a local JSON file

The API is deployed as a container on FastAPI Cloud. Containers on this kind of platform use an **ephemeral filesystem**: whenever the container sleeps and wakes back up, or is redeployed, it is rebuilt from the pushed code — anything written to disk at runtime (like a local `data/operation_log.json`) is wiped and does not persist. The database itself must live on infrastructure independent of the API container's lifecycle for logs to survive restarts.

**Supabase** (a hosted PostgreSQL database with a free tier) solves this: `usage_tracker.py` writes to and reads from an `operation_log` table over the network via the `supabase-py` client. The table lives on Supabase's own servers, entirely decoupled from the API container's lifecycle — a container restart, sleep/wake cycle, or redeploy has no effect on it.

```sql
create table operation_log (
    id             bigint generated always as identity primary key,
    timestamp      text,
    operation_type text,
    success        boolean
);
```

Row Level Security (RLS) is disabled on this table, since the log endpoint is already protected at the API layer (see below) rather than at the database layer.

### Timestamp formatting
Stored as raw ISO-8601 UTC; reformatted only at read time (`get_logs()`) into `DD/MM/YYYY hh:mm AM/PM`, localized to Africa/Cairo (e.g. `19/07/2026 11:16 PM`).

### Access control
There is no login/sign-up system in this project. `GET /logs` is gated by a single `SECRET_API_KEY`, sent as an `X-API-Key` header and compared using `secrets.compare_digest` (constant-time comparison, to avoid timing attacks). Every other endpoint is fully public.

---

## 📡 API Reference

### `GET /`
Health check / welcome message.

### `POST /predict/linear`
Predicts a price from a fully specified vehicle payload.

**Request body:** a full `CustomerData` object (`make`, `model`, `year`, `mileage`, `transmission`, `fuel_type`, `drivetrain`, `body_type`, `trim`, `engine_capacity`).

**Response (success):**
```json
{"predicted_price": 27149.06}
```

### `POST /predict/from-vin`
Predicts a price starting from a VIN plus `mileage`, `trim`, and `transmission`.

**Request body:**
```json
{
  "vin": "1FA6P8TD5M5100001",
  "mileage": 25000,
  "trim": "Sport",
  "transmission": "Manual"
}
```

**Response (success):**
```json
{
  "status": "success",
  "predicted_price": 27149.06,
  "estimated_body_type": false,
  "estimated_fuel_type": false,
  "estimated_drivetrain": false,
  "estimated_engine_capacity": false,
  "make": "Ford", "model": "Mustang", "year": 2021,
  "mileage": 25000, "transmission": "Manual",
  "fuel_type": "Gasoline", "drivetrain": "RWD",
  "body_type": "Coupe", "trim": "Sport", "engine_capacity": 2.5
}
```

**Response (make/model/year unresolved — no prediction made):**
```json
{
  "status": "incomplete",
  "predicted_price": null,
  "message": "تعذّر تحديد الماركة أو الموديل أو سنة الصنع من رقم الشاسيه بشكل موثوق...",
  "make": "...", "model": "...", "year": null, "...": "...",
  "missing_make": false, "missing_model": true, "missing_year": false
}
```

### `GET /logs`
Returns every logged operation. Requires header `X-API-Key: <SECRET_API_KEY>`.

```json
[
  {"timestamp": "19/07/2026 11:16 PM", "operation_type": "linear", "success": true},
  {"timestamp": "19/07/2026 11:15 PM", "operation_type": "from-vin", "success": true}
]
```

---

## ☁️ Deployment (FastAPI Cloud)

The API is deployed as a container on **FastAPI Cloud**, built from `pyproject.toml` / `requirements.txt` via `uv`.

- **Model artifacts** (`models/preprocessor.pkl`, `models/lin_base.pkl`) ship inside the container image.
- **Operation logs** are written to Supabase rather than local disk, so they survive container sleep/wake cycles and redeploys (see [above](#-operation-logging--supabase)).
- **Secrets** (`SECRET_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`) are configured as environment variables directly on the FastAPI Cloud project — never committed to the repo.

---

## 🐛 Deployment Issues Encountered & Fixed

Documented here for transparency and as a reference for anyone extending this project:

1. **`int()` before `round()` in `inference.py`** — an early version computed `round(int(y_pred_price[0]), 2)`, which truncates to a whole number *before* rounding, silently discarding all decimal precision from every predicted price. Fixed to `round(float(y_pred_price[0]), 2)`.
2. **`engine_capacity` not rounded to a multiple of 0.5** — the VIN decoder initially passed NHTSA's raw `DisplacementL` straight through as a continuous float, which the model never saw during training (it was only ever trained on 0.5-liter increments). Fixed with a `_round_to_nearest_half()` helper in `vin_decoder.py`.
3. **`pydantic==2.9.2` vs. `supabase==2.31.0` dependency conflict** — `supabase` (via `realtime`) requires `pydantic>=2.11.7`, which conflicted with the pinned version in `pyproject.toml`/`requirements.txt`. Fixed by relaxing the pin to `pydantic>=2.11.7,<3.0.0`.
4. **`setuptools` package-discovery failure** — with `data/` and `models/` folders present at the repo root, `setuptools`' automatic flat-layout package discovery got confused about which top-level directory was the actual Python package, and refused to build (`Multiple top-level packages discovered in a flat-layout`). Fixed by explicitly declaring the package layout in `pyproject.toml`:
   ```toml
   [tool.setuptools]
   py-modules = ["main"]
   packages = ["utils"]
   ```
5. **Local JSON log file wiped on every container restart** — see [Operation Logging & Supabase](#-operation-logging--supabase) for the full explanation and fix.

---

## 📁 Project Structure

```
.
├── main.py                      # FastAPI route definitions
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── .env.example
├── notebooks/
│   └── notebook.ipynb           # Full EDA, preprocessing, and modeling workflow
├── dataset/
│   └── vehicle_price_prediction.csv
├── assets/
│   ├── feature_importance.png
│   ├── lr_actual_vs_predicted.png
│   ├── lr_residual_plot.png
│   ├── lr_residual_distribution.png
│   ├── rf_base_actual_vs_predicted.png
│   ├── rf_base_residual_plot.png
│   ├── rf_base_residual_distribution.png
│   ├── rf_tuned_actual_vs_predicted.png
│   ├── rf_tuned_residual_plot.png
│   └── rf_tuned_residual_distribution.png
├── models/
│   ├── preprocessor.pkl         # Fitted ColumnTransformer
│   └── lin_base.pkl             # Deployed model (Linear Regression)
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── CustomerData.py
│   ├── VinRequestData.py
│   ├── vin_decoder.py
│   ├── vin_mapping.json
│   ├── vin_prediction.py
│   ├── inference.py
│   ├── error_messages.py
│   └── usage_tracker.py         # Supabase-backed operation logger
└── README.md
```

---

## 🔐 Environment Variables

| Variable | Purpose |
|---|---|
| `APP_NAME` | Displayed API title (falls back to `"Cars-Prediction"`) |
| `VERSION` | Displayed API version (falls back to `"1.0"`) |
| `SECRET_API_KEY` | Required to access `GET /logs` via the `X-API-Key` header |
| `SUPABASE_URL` | Base project URL (e.g. `https://xxxx.supabase.co`) — **without** `/rest/v1/` |
| `SUPABASE_KEY` | The `anon` / `public` API key (not `service_role`) |

See `.env.example` for a template.

---

## 🛠 Installation (Local)

```bash
git clone https://github.com/Eyad-Elghonemy/supbase-API.git
cd supbase-API
pip install -r requirements.txt
cp .env.example .env   # then fill in your own values
uvicorn main:app --reload
```

Interactive docs available at `http://127.0.0.1:8000/docs`.

---

## ⚠️ Notes & Limitations

- The **deployed** model is Linear Regression, not the more accurate Random Forest — see [Which Model Is Actually Deployed](#-which-model-is-actually-deployed).
- The tuned Random Forest was only trained once, on Colab; its metrics are documented here, but the ~7 GB model file itself could not be exported and versioned.
- Metrics are reported on `price_log`, not the raw price — keep this in mind when comparing to other price-prediction benchmarks.
- `make` and `model` are label/ordinal-encoded, not one-hot encoded, due to their high cardinality (25 and 105 categories respectively) — this can introduce artificial ordering; tree models handle this reasonably well, but it's worth revisiting with target/frequency encoding.
- There is no login/sign-up system — `GET /logs` is protected by a single shared API key, not per-user authentication.
- The VIN decoding flow depends on the free NHTSA vPIC API's uptime and data coverage; not every real-world VIN will decode to values matching this model's trained categories, by design (see [The VIN-Based Prediction Feature](#-the-vin-based-prediction-feature)).

## 🚀 Future Work

- Wire the base Random Forest model (`forest_base.pkl`) into the deployed API in place of Linear Regression, for a substantial accuracy improvement (R² 0.97 vs. ~0.91)
- Re-run and persist the tuned Random Forest in a more storage-efficient format (e.g., compressed `joblib`, or reduced `n_estimators`)
- Try target/frequency encoding for `make` and `model` instead of label encoding
- Experiment with Gradient Boosting models (XGBoost / LightGBM / CatBoost)
- Add SHAP-based feature importance for more robust, model-agnostic interpretability
- Replace the shared API-key log protection with real per-user authentication
