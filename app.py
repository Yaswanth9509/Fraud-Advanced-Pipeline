"""
app.py
All-in-one Hugging Face Space app for the fraudTrain/fraudTest schema:
  - Simulates incoming raw transactions (date, location, category, amount...)
  - Runs them through the SAME feature engineering used at training time
  - Runs hybrid inference (XGBoost + Autoencoder ensemble)
  - Stores results in a local SQLite database
  - Renders a live Streamlit dashboard

Run locally:  streamlit run app.py
Requires the 6 files produced by train_model.py to be in the same folder.
"""

import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import tensorflow as tf
from apscheduler.schedulers.background import BackgroundScheduler

from features import CATEGORY_VALUES, align_columns, engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DB_PATH = "transactions.db"
SIMULATION_INTERVAL_SECONDS = 15

_db_lock = threading.Lock()


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT (datetime('now')),
                amt REAL,
                category TEXT,
                xgb_score REAL,
                ae_score REAL,
                ae_threshold REAL,
                final_flag INTEGER,
                top_reason TEXT
            )
            """
        )
        conn.commit()


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    xgb_model = joblib.load("xgb_model.pkl")
    autoencoder = tf.keras.models.load_model("autoencoder_model.keras")
    scaler = joblib.load("scaler.pkl")
    ae_threshold = joblib.load("ae_threshold.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    freq_maps = joblib.load("category_maps.pkl")
    explainer = shap.TreeExplainer(xgb_model)
    log.info("Models loaded. Feature count: %d", len(feature_cols))
    return {
        "xgb_model": xgb_model,
        "autoencoder": autoencoder,
        "scaler": scaler,
        "ae_threshold": ae_threshold,
        "feature_cols": feature_cols,
        "freq_maps": freq_maps,
        "explainer": explainer,
    }


@st.cache_resource(show_spinner="Loading reference transaction sample...")
def load_reference_sample() -> pd.DataFrame:
    return joblib.load("reference_sample.pkl")


def generate_raw_transaction(freq_maps: dict) -> tuple:
    """
    Builds one live transaction by resampling a REAL legitimate transaction
    (preserving genuine feature correlations the model was trained on) and,
    ~8% of the time, perturbing it into a believable outlier.

    Earlier versions generated fully synthetic random values for the "normal"
    case, which don't share the real feature correlations the model learned —
    that caused a much higher false-flag rate than the intended ~8%. Resampling
    real rows and perturbing only the anomalous case fixes this.
    """
    ref = load_reference_sample()
    base = ref.sample(1).iloc[0].to_dict()

    is_anomalous = np.random.rand() < 0.08
    now = datetime.utcnow() - timedelta(minutes=np.random.randint(0, 120))

    amt = base["amt"]
    merch_lat = base["merch_lat"]
    merch_long = base["merch_long"]

    if is_anomalous:
        amt = base["amt"] * np.random.uniform(8, 25)          # unusually large charge
        merch_lat = base["lat"] + np.random.uniform(-20, 20)   # merchant far from cardholder
        merch_long = base["long"] + np.random.uniform(-20, 20)
        hour = int(np.random.choice([1, 2, 3, 4]))             # odd hour
    else:
        hour = int(np.random.randint(7, 22))

    trans_time = now.replace(hour=hour, minute=int(np.random.randint(0, 60)))

    row = {
        "trans_date_trans_time": trans_time,
        "category": base["category"],
        "amt": amt,
        "gender": base["gender"],
        "lat": base["lat"],
        "long": base["long"],
        "merch_lat": merch_lat,
        "merch_long": merch_long,
        "dob": base["dob"],
        "merchant": base["merchant"],
        "city": base["city"],
        "state": base["state"],
        "job": base["job"],
        "zip": base["zip"],
        # Identifier columns the pipeline expects but doesn't use predictively:
        "Unnamed: 0": 0, "cc_num": 0, "first": "sim", "last": "sim",
        "street": "sim", "trans_num": "sim", "unix_time": 0,
    }
    return pd.DataFrame([row]), amt


def run_inference():
    try:
        resources = load_models()
        raw_df, amt = generate_raw_transaction(resources["freq_maps"])

        feat_df, _ = engineer_features(raw_df, freq_maps=resources["freq_maps"])
        feat_df = align_columns(resources["feature_cols"], feat_df)

        scaled = resources["scaler"].transform(feat_df.values)

        xgb_prob = float(resources["xgb_model"].predict_proba(scaled)[0][1])

        recon = resources["autoencoder"].predict(scaled, verbose=0)
        ae_error = float(np.mean(np.square(scaled - recon)))
        ae_flag = ae_error > resources["ae_threshold"]

        final_flag = int((xgb_prob > 0.5) or ae_flag)

        shap_values = resources["explainer"].shap_values(scaled)
        sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        top_idx = int(np.argmax(np.abs(sv)))
        top_reason = resources["feature_cols"][top_idx] if top_idx < len(resources["feature_cols"]) else "unknown"

        with _db_lock, get_conn() as conn:
            conn.execute(
                """INSERT INTO transactions
                   (amt, category, xgb_score, ae_score, ae_threshold, final_flag, top_reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (float(amt), str(raw_df["category"].iloc[0]), xgb_prob, ae_error,
                 resources["ae_threshold"], final_flag, top_reason),
            )
            conn.commit()

    except Exception:
        log.exception("Inference cycle failed")


@st.cache_resource
def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_inference, "interval", seconds=SIMULATION_INTERVAL_SECONDS, id="fraud_sim")
    scheduler.start()
    log.info("Scheduler started: inference every %ds", SIMULATION_INTERVAL_SECONDS)
    return scheduler


def fetch_recent(limit: int = 50) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql(f"SELECT * FROM transactions ORDER BY id DESC LIMIT {limit}", conn)


# ---------------- App bootstrap ----------------
init_db()
load_models()
load_reference_sample()
start_scheduler()

st.set_page_config(page_title="Hybrid Fraud Detection", layout="wide")
st.title("Hybrid Fraud Detection — Live Dashboard")
st.caption(
    "Supervised XGBoost + unsupervised Autoencoder, ensembled. "
    f"New simulated transaction every {SIMULATION_INTERVAL_SECONDS}s."
)

df = fetch_recent(50)

col1, col2, col3 = st.columns(3)
col1.metric("Transactions (last 50)", len(df))
col2.metric("Flagged as fraud", int(df["final_flag"].sum()) if len(df) else 0)
col3.metric("Flag rate", f"{(100 * df['final_flag'].mean()):.1f}%" if len(df) else "—")

st.subheader("Recent transactions")
st.dataframe(df, use_container_width=True)

if len(df) > 1:
    chart_df = df.sort_values("id")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("XGBoost fraud probability")
        st.line_chart(chart_df.set_index("id")["xgb_score"])
    with c2:
        st.subheader("Autoencoder reconstruction error")
        st.line_chart(chart_df.set_index("id")["ae_score"])
else:
    st.info("Waiting for the first simulated transactions to arrive...")

st.button("Refresh now")