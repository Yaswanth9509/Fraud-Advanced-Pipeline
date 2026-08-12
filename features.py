"""
features.py
Shared feature-engineering logic for the fraudTrain.csv / fraudTest.csv schema
(Kaggle: kartik2112/fraud-detection).

Imported by both train_model.py (fit-time) and app.py (inference-time) so the
exact same transformation is applied in both places — this avoids "train/serve
skew," a very common real-world bug where the model is trained on features
built one way and served on features built slightly differently.
"""

import numpy as np
import pandas as pd

TARGET_COL = "is_fraud"

DROP_COLS = [
    "Unnamed: 0", "cc_num", "first", "last", "street", "trans_num",
    "unix_time", "trans_date_trans_time", "dob", "merchant",
]

FREQ_ENCODE_COLS = ["merchant_raw", "city", "state", "job", "zip"]

CATEGORY_VALUES = [
    "entertainment", "food_dining", "gas_transport", "grocery_net",
    "grocery_pos", "health_fitness", "home", "kids_pets", "misc_net",
    "misc_pos", "personal_care", "shopping_net", "shopping_pos", "travel",
]


def haversine_distance(lat1, lon1, lat2, lon2):
    """Vectorized distance (km) between cardholder location and merchant location."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def engineer_features(df: pd.DataFrame, freq_maps: dict | None = None):
    """
    Builds model-ready features from the raw Kaggle schema.

    freq_maps=None   -> fit mode: computes frequency-encoding maps from this df
                         (use only on the training set).
    freq_maps={...}  -> transform mode: reuses previously fitted maps
                         (use for test set and for live/simulated transactions).

    Returns (feature_df, freq_maps).
    """
    df = df.copy()

    dt = pd.to_datetime(df["trans_date_trans_time"])
    df["trans_hour"] = dt.dt.hour
    df["trans_day_of_week"] = dt.dt.dayofweek
    df["trans_month"] = dt.dt.month

    dob = pd.to_datetime(df["dob"])
    df["age"] = (dt - dob).dt.days // 365

    df["distance_km"] = haversine_distance(df["lat"], df["long"], df["merch_lat"], df["merch_long"])

    df["gender_male"] = (df["gender"] == "M").astype(int)

    # Fix categories to a known set so one-hot columns are identical every time,
    # regardless of what values happen to appear in a given batch.
    df["category"] = pd.Categorical(df["category"], categories=CATEGORY_VALUES)
    df = pd.get_dummies(df, columns=["category"], prefix="cat")

    df["merchant_raw"] = df["merchant"]
    compute_fresh = freq_maps is None
    if compute_fresh:
        freq_maps = {}

    for col in FREQ_ENCODE_COLS:
        if compute_fresh:
            freq_maps[col] = df[col].value_counts(normalize=True).to_dict()
        df[f"{col}_freq"] = df[col].map(freq_maps[col]).fillna(0.0)

    drop_now = [c for c in DROP_COLS if c in df.columns] + FREQ_ENCODE_COLS + \
        ["gender", "lat", "long", "merch_lat", "merch_long"]
    df = df.drop(columns=[c for c in drop_now if c in df.columns], errors="ignore")

    # pd.get_dummies produces bool columns; mixed with float columns this makes
    # DataFrame.values return an object-dtype array, which silently breaks
    # StandardScaler/XGBoost or raises confusing errors. Force everything to
    # float64 now so downstream .values calls are always safe.
    df = df.astype("float64")

    return df, freq_maps


def align_columns(reference_cols: list, other_df: pd.DataFrame) -> pd.DataFrame:
    """Ensures a dataframe has exactly `reference_cols`, in that order, filling gaps with 0."""
    other_df = other_df.copy()
    for col in reference_cols:
        if col not in other_df.columns:
            other_df[col] = 0
    extra = set(other_df.columns) - set(reference_cols)
    other_df = other_df.drop(columns=list(extra), errors="ignore")
    return other_df[reference_cols]
