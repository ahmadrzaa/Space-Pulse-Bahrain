# backend/koi_train.py — LIVE KOI download + robust tabular baseline
# Pulls the Kepler KOI "cumulative" table from NASA Exoplanet Archive at runtime (CSV via TAP).
# Robust to column differences across releases: it auto-selects whichever recommended
# numeric feature columns are present.

from __future__ import annotations
import io
import json
import os
import pickle
import time
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier


MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "koi_tabular.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "koi_metrics.json")

# Preferred KOI numeric feature candidates (we'll keep the ones that exist)
PREFERRED_FEATURES: List[str] = [
    # transit / orbital
    "koi_period",       # orbital period (days)
    "koi_time0bk",      # epoch (BKJD)
    "koi_duration",     # transit duration (hours)
    "koi_depth",        # transit depth (ppm)
    "koi_impact",       # impact parameter
    "koi_dor",          # a/R*
    # SNR / quality
    "koi_model_snr",
    "koi_snr",          # may be missing on some drops
    # planet params
    "koi_prad",         # planet radius (Earth radii)
    "koi_ror",          # Rp/R*
    # stellar params
    "koi_steff",        # stellar Teff
    "koi_slogg",        # stellar logg
    "koi_srad",         # stellar radius (solar)
]

LABEL_COL = "koi_disposition"  # {"CONFIRMED","CANDIDATE","FALSE POSITIVE"}
LABEL_MAP = {
    "CONFIRMED": "confirmed",
    "CANDIDATE": "candidate",
    "FALSE POSITIVE": "false_positive",
}
CLASS_ORDER = ["confirmed", "candidate", "false_positive"]


@dataclass
class TrainResult:
    metrics: Dict[str, Any]
    model_path: str


def _download_koi_csv() -> pd.DataFrame:
    """Live-fetch KOI Cumulative via TAP (CSV)."""
    import requests
    base = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    params = {
        "query": "select * from cumulative",  # table name per NASA TAP docs
        "format": "csv",
    }
    r = requests.get(base, params=params, headers={"Accept": "text/csv"}, timeout=120)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def _choose_features(df: pd.DataFrame) -> List[str]:
    cols = df.columns.str.lower().tolist()
    # Map to actual case by lookup
    present = []
    for name in PREFERRED_FEATURES:
        # match in case-insensitive way
        if name in df.columns:
            present.append(name)
        else:
            # try lowercase match
            try_idx = [i for i, c in enumerate(cols) if c == name.lower()]
            if try_idx:
                present.append(df.columns[try_idx[0]])
    # Deduplicate while preserving order
    seen = set()
    present = [c for c in present if not (c in seen or seen.add(c))]
    if not present:
        # As a very safe fallback, use *any* numeric koi_* columns
        numeric = [
            c for c, dt in df.dtypes.items()
            if c.startswith("koi_") and (np.issubdtype(dt, np.number))
        ]
        present = sorted(numeric)[:12]  # cap to keep model small
    return present


def _prepare(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    # Keep rows with known disposition
    if LABEL_COL not in df.columns:
        raise ValueError(f"Expected label column '{LABEL_COL}' not in table.")
    df = df[df[LABEL_COL].notna()].copy()

    # Map labels to canonical strings
    y_str = df[LABEL_COL].map(LABEL_MAP)

    # Pick feature columns that actually exist
    feat_cols = _choose_features(df)
    if not feat_cols:
        raise ValueError("No usable KOI feature columns found.")

    # Numeric, finite, fillna(0)
    X = (
        df[feat_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # Encode labels into 0..2
    y = (
        y_str.astype("category")
        .cat.set_categories(CLASS_ORDER)
        .cat.codes
        .to_numpy()
    )

    return X, y, feat_cols


def train_koi_tabular(test_size: float = 0.2, random_state: int = 42) -> TrainResult:
    df = _download_koi_csv()
    X, y, feat_cols = _prepare(df)

    # Train/val split
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Lightweight, dependable baseline
    clf = GradientBoostingClassifier(random_state=random_state)
    clf.fit(Xtr, ytr)

    proba = clf.predict_proba(Xte)

    # Metrics
    try:
        roc = roc_auc_score(yte, proba, multi_class="ovr")
    except Exception:
        roc = float("nan")
    try:
        ap = float(
            np.nanmean(
                [
                    average_precision_score((yte == i).astype(int), proba[:, i])
                    for i in range(proba.shape[1])
                ]
            )
        )
    except Exception:
        ap = float("nan")

    cm = confusion_matrix(yte, proba.argmax(1)).tolist()

    metrics = {
        "timestamp": int(time.time()),
        "samples": int(len(X)),
        "test_size": test_size,
        "classes": CLASS_ORDER,
        "roc_auc_ovr": float(roc),
        "avg_precision_macro": float(ap),
        "confusion_matrix": cm,
        "features_used": feat_cols,   # <- expose which features were present
    }

    # Persist model + metrics
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": clf, "feature_cols": feat_cols}, f)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return TrainResult(metrics=metrics, model_path=MODEL_PATH)


def load_metrics() -> Dict[str, Any] | None:
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
