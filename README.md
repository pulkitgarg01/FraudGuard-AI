# FraudGuard AI — Financial Transaction Fraud Detection and Risk Intelligence Platform

## Overview

A final-year academic prototype demonstrating an end-to-end Machine Learning pipeline integrated with a FastAPI backend and a React frontend. The platform evaluates financial transactions using a trained XGBoost model and provides a comprehensive risk intelligence dashboard, including **Explainable AI (SHAP)** — per-transaction factor explanations that show exactly which inputs drove the fraud risk assessment.

## Architecture Overview

The platform consists of three main modules:

1. **Machine Learning Pipeline (`/ml`)** — Preprocesses the IEEE-CIS Fraud Detection dataset, trains and evaluates Logistic Regression and XGBoost models, and exports the winning model (XGBoost) along with its preprocessor.
2. **Backend API (`/backend`)** — A FastAPI application that serves the ML model, computes SHAP explanations per prediction, stores transaction history in SQLite, and exposes a REST API.
3. **Frontend Dashboard (`/frontend`)** — A React 19 + TypeScript application that lets users submit transactions, view the risk assessment result, see SHAP-driven factor explanations ("Why this prediction?"), and explore analytics.

## Technology Stack

| Layer | Technologies |
|---|---|
| **Frontend** | React 19, TypeScript, Vite, React Router, Recharts |
| **Backend** | Python 3, FastAPI, SQLAlchemy, SQLite, Pydantic |
| **Machine Learning** | Scikit-Learn, XGBoost, Pandas, NumPy, Joblib |
| **Explainable AI** | SHAP (TreeExplainer) |

## Project Structure

```
major-project/
├── ml/                          # ML Pipeline & Inference
│   ├── models/                  # Saved model artifacts (.joblib)
│   │   ├── selected_model.joblib        # Deployed XGBoost model
│   │   └── selected_preprocessor.joblib # Fitted ColumnTransformer
│   ├── predict.py               # Inference + SHAP explanation engine
│   ├── preprocess.py            # Feature engineering & preprocessing pipeline
│   ├── train.py                 # Model training (LR + XGBoost)
│   ├── evaluate.py              # Model evaluation & selection
│   ├── risk_score.py            # Risk score / level computation
│   └── inspect_data.py          # Dataset inspection (Phase 0)
├── backend/                     # FastAPI Application
│   ├── app/
│   │   ├── routers/prediction.py  # POST /predict endpoint
│   │   ├── schemas.py             # Pydantic models incl. SHAP Explanation schema
│   │   ├── services/              # Prediction service layer
│   │   ├── models.py              # SQLAlchemy ORM models
│   │   └── database.py            # SQLite connection
│   ├── requirements.txt
│   └── fraud_detection.db       # SQLite database (auto-created)
├── frontend/                    # React Dashboard
│   ├── src/
│   │   ├── pages/Assessment.tsx   # Transaction input form
│   │   ├── components/ui/ResultPanel.tsx  # Risk result + SHAP panel
│   │   ├── types/index.ts         # TypeScript interfaces
│   │   └── api/fraud.ts           # API client
│   └── package.json
└── README.md
```

## ML Model Details

| Item | Value |
|---|---|
| Dataset | IEEE-CIS Fraud Detection (Kaggle) |
| Training samples | ~473,000 transactions |
| Test samples | ~118,000 transactions |
| Fraud rate | 3.50% |
| Selected model | XGBoost (`n_estimators=100, max_depth=4, lr=0.1`) |
| Input features | 14 raw → 31 processed (after one-hot encoding) |

**Model evaluation on test set:**

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.0849 | 0.6363 | 0.1498 | 0.7521 |
| **XGBoost (selected)** | **0.1293** | **0.7234** | **0.2195** | **0.8587** |

XGBoost was selected based on higher ROC-AUC, which best reflects performance on the heavily imbalanced dataset.

## How Prediction Works

1. User submits a transaction form on the frontend → `POST /predict`
2. FastAPI backend passes raw fields to `ml/predict.py`
3. Feature engineering: derive `tx_hour`, `tx_day` from `TransactionDT`; compute `log1p(TransactionAmt)`; coarsen email domain to 6 groups
4. `selected_preprocessor.joblib` (ColumnTransformer) transforms 14 features → 31 processed columns (StandardScaler + OneHotEncoder + passthrough)
5. `selected_model.joblib` (XGBoost) outputs fraud probability and binary prediction
6. Risk score (0–100) and risk level (`LOW` / `MEDIUM` / `HIGH`) computed from probability
7. **SHAP TreeExplainer** generates one SHAP value per processed feature; values are aggregated into 12 user-facing groups and ranked by contribution magnitude
8. Result stored in SQLite; full response (including SHAP explanation) returned to frontend
9. Frontend displays verdict, risk score gauge, fraud probability bar, and **"Why This Prediction?"** explanation panel

## Explainable AI — SHAP

Every prediction includes a SHAP-based explanation showing which input features contributed most to the fraud risk assessment:

- **Increasing Risk** — factors that pushed the model toward a higher fraud prediction
- **Reducing Risk** — factors that pushed the model toward a lower fraud prediction
- Bars show proportional relative contribution among the displayed factors
- Explanations are dynamically generated per transaction — not hardcoded

> **Honest disclaimer (shown in the UI):** SHAP values describe why the XGBoost **model** produced its output — not why fraud actually occurred in reality.

## Installation & Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- macOS users: XGBoost requires `libomp` — install via Homebrew:
  ```bash
  brew install libomp
  ```

### 1. Backend Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** `requirements.txt` includes `shap>=0.44.0`. SHAP will be installed automatically.

### 2. Frontend Setup

```bash
cd frontend
npm install
```

## Running the Application

### Start the Backend

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload
```

- API: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`

### Start the Frontend

Open a **new terminal**:

```bash
cd frontend
npm run dev
```

- Dashboard: `http://localhost:5173`

## Example API Request

**POST** `/predict`

```json
{
  "TransactionAmt": 450.0,
  "TransactionDT": 86400,
  "ProductCD": "C",
  "card1": 12345,
  "card2": 111,
  "card4": "mastercard",
  "card6": "credit",
  "addr1": 999,
  "C1": 4,
  "C5": 0,
  "P_emaildomain": "gmail.com",
  "id_present": 1
}
```

**Response**

```json
{
  "transaction_id": "TXN-94210404",
  "prediction": 1,
  "prediction_label": "Fraud",
  "fraud_probability": 0.9077,
  "risk_score": 91,
  "risk_level": "HIGH",
  "model_used": "XGBoost",
  "explanation": {
    "available": true,
    "top_risk_factors": [
      { "feature": "Addresses Linked to Card", "direction": "increases_risk", "relative_impact": 1.0 },
      { "feature": "Card Account Tier",        "direction": "increases_risk", "relative_impact": 0.72 },
      { "feature": "Card Type",                "direction": "increases_risk", "relative_impact": 0.54 }
    ],
    "top_protective_factors": [
      { "feature": "Recent Transaction Velocity", "direction": "reduces_risk", "relative_impact": 1.0 },
      { "feature": "Email Provider",              "direction": "reduces_risk", "relative_impact": 0.63 },
      { "feature": "Transaction Amount",          "direction": "reduces_risk", "relative_impact": 0.41 }
    ]
  }
}
```

## Dataset

- **Source:** [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)
- **Files used:** `train_transaction.csv` (590K rows), `train_identity.csv` (144K rows)
- **Target variable:** `isFraud` (binary: 0 = Legitimate, 1 = Fraud)
- Raw data files are **not included** in this repository (too large). Download from Kaggle and place in `data/raw/` before running `ml/preprocess.py`.
