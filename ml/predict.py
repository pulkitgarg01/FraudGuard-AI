"""
predict.py
==========
Inference module: accepts one raw transaction as a Python dict and returns
a fraud prediction, probability, risk score, risk level, and SHAP explanation.

This is the file that the FastAPI backend calls.

It loads:
  - selected_preprocessor.joblib  (fitted ColumnTransformer from preprocess.py)
  - selected_model.joblib         (the winning XGBClassifier from evaluate.py)

SHAP:
  - A TreeExplainer is created once when the model is first loaded.
  - For each prediction, SHAP values are computed on the preprocessed (1, 31)
    array — the same representation XGBoost uses internally.
  - SHAP semantics verified: positive value → pushes toward fraud class (class 1),
    negative value → pushes toward legitimate class (class 0).
  - One-hot and engineered features are aggregated back to user-facing groups.
"""

import os
import time
import numpy as np
import pandas as pd
import joblib

from preprocess import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BINARY_FEATURES,
    coarsen_email_domain,
)
from risk_score import compute_risk

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_HERE, "models")

SELECTED_MODEL_PATH = os.path.join(MODELS_DIR, "selected_model.joblib")
SELECTED_PREP_PATH  = os.path.join(MODELS_DIR, "selected_preprocessor.joblib")

# Module-level singletons — loaded once at first prediction, reused thereafter.
_model        = None
_preprocessor = None
_explainer    = None   # shap.TreeExplainer instance


# ── Feature group mapping: 31 processed columns → 9 user-facing groups ────────
#
# SHAP produces one value per processed column (31 total).
# These groups aggregate related columns back to the feature the user actually
# provided. One-hot columns for the same source feature are summed together.
#
# Labelling philosophy:
#   - "TransactionAmt" and "amt_log" both come from the one TransactionAmt input.
#   - "tx_hour" and "tx_day" both come from TransactionDT (hidden from frontend).
#     They are grouped under "Transaction Time Pattern".
#   - card1 and card2 are anonymised numeric identifiers in the IEEE-CIS dataset.
#     They are labelled neutrally as "Card Identifier 1" and "Card Identifier 2".
#   - addr1 is a billing/address region code — labelled "Billing Region Code".
#   - C1 and C5 are anonymised count-type features — labelled neutrally.
#   - id_present is the identity-record flag — labelled "Identity Record Present".
#
# The label is what the user sees; the columns list is the set of processed
# feature names (from preprocessor.get_feature_names_out()) to aggregate.

FEATURE_GROUPS = [
    {
        # Form label: "Transaction Amount (in USD)"
        "label":   "Transaction Amount",
        "columns": ["num__TransactionAmt", "num__amt_log"],
    },
    {
        # Derived from TransactionDT — not a visible form field, kept descriptive
        "label":   "Transaction Time of Day",
        "columns": ["num__tx_hour", "num__tx_day"],
    },
    {
        # Form label: "Card Account Tier"
        "label":   "Card Account Tier",
        "columns": ["num__card1"],
    },
    {
        # Form label: "Issuing Bank"
        "label":   "Issuing Bank",
        "columns": ["num__card2"],
    },
    {
        # Form label: "Billing Region / Location"
        "label":   "Billing Region",
        "columns": ["num__addr1"],
    },
    {
        # Form label: "Addresses Linked to Card"
        "label":   "Addresses Linked to Card",
        "columns": ["num__C1"],
    },
    {
        # Form label: "Recent Transaction Velocity"
        "label":   "Recent Transaction Velocity",
        "columns": ["num__C5"],
    },
    {
        # Form label: "Product / Service Type"
        "label":   "Product / Service Type",
        "columns": [
            "cat__ProductCD_C", "cat__ProductCD_H", "cat__ProductCD_R",
            "cat__ProductCD_S", "cat__ProductCD_W",
        ],
    },
    {
        # Form label: "Card Network"
        "label":   "Card Network",
        "columns": [
            "cat__card4_american express", "cat__card4_discover",
            "cat__card4_mastercard",        "cat__card4_missing",
            "cat__card4_visa",
        ],
    },
    {
        # Form label: "Card Type"
        "label":   "Card Type",
        "columns": [
            "cat__card6_charge card", "cat__card6_credit",
            "cat__card6_debit",       "cat__card6_debit or credit",
            "cat__card6_missing",
        ],
    },
    {
        # Form label: "Purchaser's Email Provider"
        "label":   "Email Provider",
        "columns": [
            "cat__P_emaildomain_anonymous", "cat__P_emaildomain_gmail",
            "cat__P_emaildomain_microsoft", "cat__P_emaildomain_missing",
            "cat__P_emaildomain_other",      "cat__P_emaildomain_yahoo",
        ],
    },
    {
        # Form label: "Identity Verified?"
        "label":   "Identity Verified",
        "columns": ["bin__id_present"],
    },
]

# Pre-build a lookup: processed_column_name → group index, for fast aggregation.
_COL_TO_GROUP: dict[str, int] = {}
for _g_idx, _g in enumerate(FEATURE_GROUPS):
    for _col in _g["columns"]:
        _COL_TO_GROUP[_col] = _g_idx


# ── Load artifacts (once, lazily) ──────────────────────────────────────────────
def _load_artifacts():
    """Load model, preprocessor, and SHAP explainer from disk on first call."""
    global _model, _preprocessor, _explainer
    if _model is not None:
        return   # already loaded

    _model        = joblib.load(SELECTED_MODEL_PATH)
    _preprocessor = joblib.load(SELECTED_PREP_PATH)
    print(f"[predict.py] Model loaded: {SELECTED_MODEL_PATH}")
    print(f"[predict.py] Preprocessor loaded: {SELECTED_PREP_PATH}")

    # Build SHAP TreeExplainer once.
    # TreeExplainer is the native, fastest explainer for XGBoost tree models.
    # It does NOT require background data for XGBoost — it uses the tree structure
    # directly to compute exact (not approximate) SHAP values.
    try:
        import shap as _shap
        _explainer = _shap.TreeExplainer(_model)
        print("[predict.py] SHAP TreeExplainer created OK")
    except ImportError:
        _explainer = None
        print("[predict.py] WARNING: shap not installed — explanations disabled")
    except Exception as e:
        _explainer = None
        print(f"[predict.py] WARNING: SHAP explainer creation failed: {e}")


# ── Feature engineering for a single transaction ──────────────────────────────
def _engineer_single(transaction: dict) -> pd.DataFrame:
    """
    Apply the same feature engineering steps used during training
    to a single raw transaction dict.
    """
    row = dict(transaction)  # copy so we don't modify the caller's dict

    # Time features derived from TransactionDT (elapsed seconds)
    dt = row.get("TransactionDT", 0)
    row["tx_hour"] = int((dt // 3600) % 24)
    row["tx_day"]  = int((dt // 86400) % 7)

    # Log amount — reduces right skew in the model's view of amount
    amt = row.get("TransactionAmt", 0.0)
    row["amt_log"] = float(np.log1p(amt))

    # Email domain coarsening (59 → 6 groups)
    raw_domain = row.get("P_emaildomain", None)
    row["P_emaildomain"] = coarsen_email_domain(pd.Series([raw_domain])).iloc[0]

    # id_present: 1 if identity record exists, 0 if not provided
    if "id_present" not in row:
        row["id_present"] = 0

    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
    return pd.DataFrame([row])[all_features]


# ── SHAP aggregation: 31 values → user-facing groups ──────────────────────────
def _build_explanation(
    shap_values: np.ndarray,
    feature_names: list[str],
    transaction: dict,
    max_factors: int = 3,
) -> dict:
    """
    Aggregate 31 raw SHAP values into human-readable factor lists.

    SHAP semantics (verified for this model/version):
      - Positive SHAP value → pushes the model's output toward FRAUD (class 1)
      - Negative SHAP value → pushes the model's output toward LEGITIMATE (class 0)

    Aggregation:
      - For each group, sum the SHAP values of all its member columns.
      - The sign of the aggregate determines direction.
      - Zero/near-zero groups are excluded from the top-N display.

    Returns a dict with 'available', 'top_risk_factors', 'top_protective_factors'.
    """
    # Build name → index map for this specific model's feature names
    name_to_idx = {n: i for i, n in enumerate(feature_names)}

    # Aggregate SHAP values per group
    group_totals = []
    for g in FEATURE_GROUPS:
        total = 0.0
        for col in g["columns"]:
            idx = name_to_idx.get(col)
            if idx is not None:
                val = float(shap_values[idx])
                if not np.isnan(val):
                    total += val
        group_totals.append((g["label"], total))

    # Separate by direction, sort by absolute magnitude descending
    increasing = [(lbl, v) for lbl, v in group_totals if v > 0]
    reducing   = [(lbl, v) for lbl, v in group_totals if v < 0]

    increasing.sort(key=lambda x: x[1], reverse=True)   # highest positive first
    reducing.sort(key=lambda x: x[1])                    # most negative first

    # Top-N from each side
    top_inc = increasing[:max_factors]
    top_red = reducing[:max_factors]

    # Compute relative impact (0.0–1.0) within each side's displayed list
    # so the frontend can draw proportional bars without misleading percentages.
    def _to_factors(pairs: list, direction: str) -> list:
        if not pairs:
            return []
        max_mag = max(abs(v) for _, v in pairs)
        result = []
        for label, shap_val in pairs:
            result.append({
                "feature":         label,
                "direction":       direction,
                "relative_impact": round(abs(shap_val) / max_mag, 4) if max_mag else 0.0,
            })
        return result

    return {
        "available":              True,
        "top_risk_factors":       _to_factors(top_inc, "increases_risk"),
        "top_protective_factors": _to_factors(top_red, "reduces_risk"),
    }


# ── Public prediction function ─────────────────────────────────────────────────
def predict_transaction(transaction: dict) -> dict:
    """
    Accept one raw transaction dict and return a prediction with SHAP explanation.

    Parameters
    ----------
    transaction : dict
        Raw transaction data. Expected keys: TransactionAmt, TransactionDT,
        ProductCD, card1, card2, card4, card6, addr1, C1, C5,
        P_emaildomain, id_present (optional).

    Returns
    -------
    dict with keys:
        prediction        : int    (0 = Legitimate, 1 = Fraud)
        fraud_probability : float  (0.0 to 1.0)
        risk_score        : int    (0 to 100)
        risk_level        : str    ('LOW', 'MEDIUM', or 'HIGH')
        explanation       : dict   (SHAP-based factor list, or unavailable)
    """
    _load_artifacts()

    # Feature engineering → preprocessing (same as training time)
    df_row = _engineer_single(transaction)
    X_proc = _preprocessor.transform(df_row)  # shape (1, 31), float64, dense

    # Core prediction
    prediction        = int(_model.predict(X_proc)[0])
    fraud_probability = float(_model.predict_proba(X_proc)[0, 1])
    risk              = compute_risk(fraud_probability)

    # SHAP explanation
    explanation = {"available": False}
    if _explainer is not None:
        try:
            t0 = time.perf_counter()
            shap_values = _explainer.shap_values(X_proc)   # shape (1, 31)
            elapsed_ms  = (time.perf_counter() - t0) * 1000

            feature_names = list(_preprocessor.get_feature_names_out())
            sv_row        = shap_values[0]   # shape (31,)

            explanation = _build_explanation(sv_row, feature_names, transaction)
            explanation["_shap_ms"] = round(elapsed_ms, 1)  # internal timing

        except Exception as e:
            explanation = {"available": False, "error": str(e)}

    return {
        "prediction":        prediction,
        "fraud_probability": round(fraud_probability, 4),
        "risk_score":        risk["risk_score"],
        "risk_level":        risk["risk_level"],
        "explanation":       explanation,
    }


# ── Demo when run directly ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    samples = {
        "LEGIT": {
            "TransactionAmt": 50.00,
            "TransactionDT":  86400,
            "ProductCD":      "W",
            "card1":          10001,
            "card2":          321.0,
            "card4":          "visa",
            "card6":          "debit",
            "addr1":          100.0,
            "C1":             1.0,
            "C5":             0.0,
            "P_emaildomain":  "gmail.com",
            "id_present":     0,
        },
        "FRAUD": {
            "TransactionAmt": 2500.00,
            "TransactionDT":  7200,
            "ProductCD":      "R",
            "card1":          99999,
            "card2":          111.0,
            "card4":          "discover",
            "card6":          "credit",
            "addr1":          999.0,
            "C1":             15.0,
            "C5":             15.0,
            "P_emaildomain":  "protonmail.com",
            "id_present":     1,
        },
    }

    for label, txn in samples.items():
        print(f"\n{'='*60}")
        print(f"  {label} transaction")
        print('='*60)
        result = predict_transaction(txn)
        print(json.dumps(result, indent=2))
