"""
train_model.py
Trains a hybrid fraud detection system on the Kaggle "Fraud Detection"
dataset (kartik2112/fraud-detection) — fraudTrain.csv + fraudTest.csv.

  1. Supervised XGBoost classifier (with SMOTE oversampling)
  2. Unsupervised TensorFlow autoencoder (trained only on legitimate transactions)

Run locally:  python train_model.py
Requires: data/fraudTrain.csv, data/fraudTest.csv

Outputs (used later by app.py):
  xgb_model.pkl
  autoencoder_model.keras
  scaler.pkl
  ae_threshold.pkl
  feature_columns.pkl
  category_maps.pkl
"""

import json
import logging
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import callbacks, layers, models

from features import TARGET_COL, align_columns, engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

TRAIN_PATH = "data/fraudTrain.csv"
TEST_PATH = "data/fraudTest.csv"
RANDOM_STATE = 42


def load_raw(path: str):
    import pandas as pd
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        log.error("File not found: %s. Place fraudTrain.csv / fraudTest.csv under data/.", path)
        sys.exit(1)
    if TARGET_COL not in df.columns:
        log.error("Expected target column '%s' not found in %s.", TARGET_COL, path)
        sys.exit(1)
    return df


def train_xgb(X_train, y_train, X_test, y_test) -> xgb.XGBClassifier:
    log.info("Balancing training set with SMOTE...")
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    log.info("Post-SMOTE class balance: %s", np.bincount(y_res))

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_res, y_res, eval_set=[(X_test, y_test)], verbose=False)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)
    auc = roc_auc_score(y_test, probs)
    log.info("XGBoost ROC-AUC on test set: %.4f", auc)
    log.info("\n%s", classification_report(y_test, preds, digits=4))
    return model


def train_autoencoder(X_train_legit: np.ndarray, input_dim: int):
    log.info("Training autoencoder on %d legitimate transactions...", len(X_train_legit))
    autoencoder = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(16, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(input_dim, activation="linear"),
    ])
    autoencoder.compile(optimizer="adam", loss="mse")

    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    autoencoder.fit(
        X_train_legit, X_train_legit,
        epochs=50, batch_size=256, validation_split=0.1,
        callbacks=[early_stop], verbose=0,
    )
    return autoencoder


def compute_threshold(autoencoder, X_legit: np.ndarray, percentile: float = 95.0) -> float:
    recon = autoencoder.predict(X_legit, verbose=0)
    mse = np.mean(np.square(X_legit - recon), axis=1)
    threshold = float(np.percentile(mse, percentile))
    log.info("Anomaly threshold (%.0fth percentile): %.6f", percentile, threshold)
    return threshold


def main():
    train_raw = load_raw(TRAIN_PATH)
    test_raw = load_raw(TEST_PATH)
    log.info("Train rows: %d | Test rows: %d | Fraud rate (train): %.4f%%",
              len(train_raw), len(test_raw), 100 * train_raw[TARGET_COL].mean())

    y_train = train_raw[TARGET_COL].values
    y_test = test_raw[TARGET_COL].values

    train_feat, freq_maps = engineer_features(train_raw.drop(columns=[TARGET_COL]))
    test_feat, _ = engineer_features(test_raw.drop(columns=[TARGET_COL]), freq_maps=freq_maps)
    test_feat = align_columns(list(train_feat.columns), test_feat)

    feature_cols = list(train_feat.columns)
    log.info("Engineered feature count: %d", len(feature_cols))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_feat.values)
    X_test = scaler.transform(test_feat.values)

    xgb_model = train_xgb(X_train, y_train, X_test, y_test)

    X_train_legit = X_train[y_train == 0]
    autoencoder = train_autoencoder(X_train_legit, input_dim=X_train.shape[1])
    threshold = compute_threshold(autoencoder, X_train_legit, percentile=95.0)

    X_test_recon = autoencoder.predict(X_test, verbose=0)
    test_mse = np.mean(np.square(X_test - X_test_recon), axis=1)
    ae_auc = roc_auc_score(y_test, test_mse)
    log.info("Autoencoder reconstruction-error ROC-AUC on test set: %.4f", ae_auc)

    joblib.dump(xgb_model, "xgb_model.pkl")
    autoencoder.save("autoencoder_model.keras")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(threshold, "ae_threshold.pkl")
    joblib.dump(feature_cols, "feature_columns.pkl")
    joblib.dump(freq_maps, "category_maps.pkl")

    metrics = {
        "xgb_roc_auc": float(roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])),
        "autoencoder_roc_auc": float(ae_auc),
        "ae_threshold_percentile": 95.0,
        "ae_threshold_value": threshold,
        "n_features": len(feature_cols),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open("training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("Done. Saved: xgb_model.pkl, autoencoder_model.keras, scaler.pkl, "
              "ae_threshold.pkl, feature_columns.pkl, category_maps.pkl, training_metrics.json")


if __name__ == "__main__":
    main()
