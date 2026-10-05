"""
train_model.py
Trains a Random Forest regressor on song_data.csv and saves:
  - model/rf_model.pkl   (trained model)
  - model/scaler.pkl     (feature scaler)
  - model/feature_names.pkl (ordered feature list)
  - model/metrics.json   (MAE, RMSE, R²)
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "..", "data", "song_data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "..", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Feature columns ──────────────────────────────────────────────────────────
FEATURES = [
    "song_duration_ms",
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "key",
    "liveness",
    "loudness",
    "audio_mode",
    "speechiness",
    "tempo",
    "time_signature",
    "audio_valence",
]
TARGET = "song_popularity"


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=FEATURES + [TARGET])
    # Remove obvious outliers (popularity outside 0-100)
    df = df[(df[TARGET] >= 0) & (df[TARGET] <= 100)]
    return df


def train():
    print("Loading data …")
    df = load_and_clean(DATA_PATH)
    print(f"  Rows after cleaning: {len(df):,}")

    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Try multiple models, pick the best ───────────────────────────────────
    candidates = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=42
        ),
        "Ridge": Ridge(alpha=1.0),
    }

    best_name, best_model, best_r2 = None, None, -999
    results = {}

    for name, mdl in candidates.items():
        mdl.fit(X_train_s, y_train)
        preds = mdl.predict(X_test_s)
        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        results[name] = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}
        print(f"  {name:20s}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
        if r2 > best_r2:
            best_r2, best_name, best_model = r2, name, mdl

    print(f"\n  Best model: {best_name}  (R²={best_r2:.4f})")

    # ── Feature importances (for RF / GB) ────────────────────────────────────
    importances = {}
    if hasattr(best_model, "feature_importances_"):
        for fname, imp in zip(FEATURES, best_model.feature_importances_):
            importances[fname] = round(float(imp), 6)

    # ── Persist artefacts ────────────────────────────────────────────────────
    with open(os.path.join(MODEL_DIR, "rf_model.pkl"),       "wb") as f:
        pickle.dump(best_model, f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"),         "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "feature_names.pkl"),  "wb") as f:
        pickle.dump(FEATURES, f)

    metrics = {
        "best_model": best_name,
        "test_metrics": results[best_name],
        "all_models": results,
        "feature_importances": importances,
        "train_rows": int(len(X_train)),
        "test_rows":  int(len(X_test)),
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("  Artefacts saved to model/")
    print(json.dumps(metrics["test_metrics"], indent=2))


if __name__ == "__main__":
    train()
