"""
app.py
Real-time fraud detector for the fraudTrain/fraudTest schema. Single
process, Neon Postgres backend, and TWO interchangeable ingestion modes
chosen automatically at startup by get_pipeline_mode():

  - "kafka"      -- a Kafka broker is reachable. A daemon thread publishes
                    simulated transactions to the 'transactions-stream'
                    topic; an APScheduler job consumes them back.
  - "in-process" -- no broker reachable. The same simulation logic runs
                    inline and hands transactions straight to the scorer.

Kafka needs a broker and there is no permanently-free hosted Kafka, so the
public Streamlit Community Cloud demo has none. Without the fallback it
would render a dashboard that never receives a transaction. The active
mode is displayed prominently in the UI so the two are never confused.

Everything after ingestion is identical in both modes:
  - The SAME feature engineering used at training time (features.py)
  - Hybrid inference (XGBoost + Autoencoder ensemble) + SHAP top reason
  - Results stored in Neon Postgres (persists across app restarts)
  - A live dashboard: ingestion mode, consumer lag (Kafka mode only),
    transactions/sec, and pipeline uptime

The producer used to be a separate process (producer.py) that had to be
started by hand alongside this one. Streamlit Community Cloud only ever
runs this single file, so the same publishing logic now also runs here as
a background thread -- opening the app's URL is all it takes to start the
whole pipeline. producer.py still exists as a standalone copy of that
logic, but it is NOT needed: running it while this app is up simply adds
a second publisher to the same topic (roughly doubling the transaction
rate), which is only useful as an extra load generator.

Run locally:
    docker compose up -d   # or point KAFKA_BOOTSTRAP_SERVERS at a
                            # hosted broker in .streamlit/secrets.toml
    streamlit run app.py   # publishes AND consumes -- nothing else needed

Requires:
  - The 6 files produced by train_model.py, in the same folder
  - reference_sample.pkl (the simulator resamples from this)
  - A DATABASE_URL secret pointing at a Neon Postgres database
    (locally: .streamlit/secrets.toml; on Streamlit Cloud: Settings -> Secrets)
  - A reachable Kafka broker. Locally this defaults to Docker Kafka on
    localhost:9092 with no auth (see docker-compose.yml). For a broker
    reachable from Streamlit Cloud (e.g. Confluent Cloud), set
    KAFKA_BOOTSTRAP_SERVERS, KAFKA_API_KEY, and KAFKA_API_SECRET as
    secrets -- SASL_SSL auth is used automatically whenever KAFKA_API_KEY
    is present, so the same code runs against both.
"""

import collections
import json
import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import shap
import streamlit as st
import tensorflow as tf
from apscheduler.schedulers.background import BackgroundScheduler
from kafka import KafkaConsumer, KafkaProducer, TopicPartition

from features import CATEGORY_VALUES, align_columns, engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 5          # how often the scheduler drains the Kafka topic
PRODUCE_INTERVAL_SECONDS = 5       # how often the in-app producer thread publishes
KAFKA_BOOTSTRAP_SERVERS_DEFAULT = "localhost:9092"     # local Docker Kafka fallback
KAFKA_TOPIC = "transactions-stream"

_db_lock = threading.Lock()


def _kafka_connection_kwargs() -> dict:
    """
    Connection kwargs shared by every Kafka client in this file (the
    in-app producer thread, the main consumer, the lag-check consumer and
    admin client). KAFKA_BOOTSTRAP_SERVERS is read from secrets when
    present -- e.g. a Confluent Cloud bootstrap URL -- falling back to
    local Docker Kafka on localhost:9092 otherwise, so `docker compose up
    -d` + `streamlit run app.py` keeps working with no secrets changes.
    SASL_SSL auth (KAFKA_API_KEY / KAFKA_API_SECRET) is added only when a
    KAFKA_API_KEY secret is actually present, since local Docker Kafka
    runs PLAINTEXT with no auth at all and would reject SASL_SSL.
    """
    kwargs = {"bootstrap_servers": st.secrets.get("KAFKA_BOOTSTRAP_SERVERS", KAFKA_BOOTSTRAP_SERVERS_DEFAULT)}
    api_key = st.secrets.get("KAFKA_API_KEY")
    if api_key:
        kwargs.update(
            security_protocol="SASL_SSL",
            sasl_mechanism="PLAIN",
            sasl_plain_username=api_key,
            sasl_plain_password=st.secrets.get("KAFKA_API_SECRET", ""),
        )
    return kwargs

STREAM_RATE_WINDOW_SECONDS = 30


@st.cache_resource
def get_stream_state() -> dict:
    """
    Rolling window of local receive-times for consumed messages (messages/sec)
    plus the producer's self-reported start time (producer uptime). Both are
    derived purely from what's already flowing through the topic -- no
    separate heartbeat channel needed.

    Deliberately @st.cache_resource rather than a plain module-level global:
    Streamlit re-executes this whole script top-to-bottom on every rerun, so
    a plain global gets silently reinitialized each time. The background
    scheduler thread, though, is bound forever to the FIRST execution's
    run_inference closure (start_scheduler() is itself cached and only ever
    registers that one job). A plain global would mean the thread keeps
    writing into an orphaned first-run copy that later reruns' rendering
    code never sees -- msgs/sec would silently read 0 forever. Routing both
    the writer (_record_stream_stat, called from the thread) and the reader
    (get_stream_metrics, called from the render path) through this same
    cached singleton keeps them talking about the same object regardless of
    which rerun's namespace is asking.
    """
    return {
        "lock": threading.Lock(),
        "recent_receive_times": collections.deque(maxlen=500),
        "producer_started_at": None,
    }


def _record_stream_stat(row: dict) -> None:
    state = get_stream_state()
    with state["lock"]:
        state["recent_receive_times"].append(time.time())
        started_at = row.get("producer_started_at")
        if started_at:
            state["producer_started_at"] = started_at


def get_stream_metrics() -> tuple:
    """Returns (messages_per_sec, producer_uptime_str) computed from the
    cached stream state. Uptime is None if no messages have been consumed
    yet (or the producer predates this field being added)."""
    state = get_stream_state()
    with state["lock"]:
        recent = list(state["recent_receive_times"])
        started_at = state["producer_started_at"]

    now = time.time()
    in_window = [t for t in recent if now - t <= STREAM_RATE_WINDOW_SECONDS]
    msgs_per_sec = len(in_window) / STREAM_RATE_WINDOW_SECONDS if in_window else 0.0

    uptime_str = None
    if started_at:
        started = datetime.fromisoformat(started_at)
        elapsed = datetime.now(timezone.utc) - started
        total_seconds = int(elapsed.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours}h {minutes}m {seconds}s" if hours else f"{minutes}m {seconds}s"

    return msgs_per_sec, uptime_str


def get_database_url() -> str:
    """Reads the Neon connection string from Streamlit secrets."""
    try:
        return st.secrets["DATABASE_URL"]
    except (KeyError, FileNotFoundError):
        log.error(
            "DATABASE_URL not found in Streamlit secrets. "
            "Add it under Settings -> Secrets on Streamlit Cloud, "
            "or in .streamlit/secrets.toml for local runs."
        )
        st.error(
            "Database not configured. Add DATABASE_URL to Streamlit secrets "
            "(Settings -> Secrets on Streamlit Cloud)."
        )
        st.stop()


@contextmanager
def get_conn():
    conn = psycopg2.connect(get_database_url(), connect_timeout=10)
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id SERIAL PRIMARY KEY,
                ts TIMESTAMP DEFAULT NOW(),
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


@st.cache_resource(show_spinner="Connecting to Kafka...")
def get_consumer() -> KafkaConsumer:
    """
    One shared consumer instance for the app's lifetime. auto_offset_reset
    'latest' means a freshly-started app only sees NEW transactions
    published after it connects, not the whole topic history — matches
    the old behavior of only ever showing freshly-generated transactions.
    """
    return KafkaConsumer(
        KAFKA_TOPIC,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
        consumer_timeout_ms=1000,  # poll() below returns promptly if nothing's new
        group_id="fraud-dashboard-consumer",
        **_kafka_connection_kwargs(),
    )


def get_consumer_lag() -> int:
    """
    Total unread messages across all partitions of the topic.

    IMPORTANT: this uses its OWN short-lived consumer, separate from the
    one run_inference() polls in the background scheduler thread.
    KafkaConsumer instances are NOT thread-safe - sharing one consumer
    between the APScheduler background thread (run_inference) and the
    Streamlit main thread (this function, called on every page rerun)
    caused silent corruption: poll() calls from two threads collided,
    run_inference()'s broad try/except swallowed the resulting errors,
    and APScheduler kept reporting "executed successfully" even though
    no new rows were being written to Postgres. A dedicated consumer,
    created and torn down within this single function call, avoids any
    cross-thread access entirely.
    """
    try:
        lag_consumer = KafkaConsumer(
            consumer_timeout_ms=1000,
            # A distinct, throwaway group_id so this never competes with
            # the real inference consumer's group for partition assignment
            # or offset commits.
            group_id="fraud-dashboard-lag-check",
            **_kafka_connection_kwargs(),
        )
        lag_consumer.assign([TopicPartition(KAFKA_TOPIC, p) for p in lag_consumer.partitions_for_topic(KAFKA_TOPIC) or []])
        partitions = lag_consumer.assignment()
        if not partitions:
            lag_consumer.close()
            return 0
        lag_consumer.seek_to_end(*partitions)
        end_offsets = {tp: lag_consumer.position(tp) for tp in partitions}

        # Compare against the REAL inference consumer group's committed offsets
        from kafka import KafkaAdminClient
        admin = KafkaAdminClient(**_kafka_connection_kwargs())
        committed = admin.list_group_offsets("fraud-dashboard-consumer").get("fraud-dashboard-consumer", {})
        admin.close()

        lag = 0
        for tp in partitions:
            committed_offset = committed.get(tp)
            current = committed_offset.offset if committed_offset else 0
            lag += max(end_offsets[tp] - current, 0)

        lag_consumer.close()
        return lag
    except Exception:
        log.exception("Could not compute consumer lag")
        return -1


@st.cache_resource
def load_reference_sample() -> pd.DataFrame:
    return joblib.load("reference_sample.pkl")


def generate_raw_transaction(ref: pd.DataFrame) -> dict:
    """
    Resamples a real legitimate transaction and, ~8% of the time, perturbs
    it into a believable outlier. Identical logic to producer.py's version
    (kept as a separate standalone file for local dev) -- returns a plain
    JSON-serializable dict since this travels over Kafka either way.
    """
    base = ref.sample(1).iloc[0].to_dict()

    is_anomalous = np.random.rand() < 0.08
    now = datetime.now(timezone.utc) - timedelta(minutes=np.random.randint(0, 120))

    amt = base["amt"]
    merch_lat = base["merch_lat"]
    merch_long = base["merch_long"]

    if is_anomalous:
        amt = base["amt"] * np.random.uniform(8, 25)
        merch_lat = base["lat"] + np.random.uniform(-20, 20)
        merch_long = base["long"] + np.random.uniform(-20, 20)
        hour = int(np.random.choice([1, 2, 3, 4]))
    else:
        orig_time = pd.to_datetime(base["trans_date_trans_time"])
        hour = int(orig_time.hour)

    trans_time = now.replace(hour=hour, minute=int(np.random.randint(0, 60)))
    trans_time = trans_time.replace(tzinfo=None)  # keep naive, same reason as original code

    return {
        "trans_date_trans_time": trans_time.isoformat(),
        "category": base["category"],
        "amt": float(amt),
        "gender": base["gender"],
        "lat": float(base["lat"]),
        "long": float(base["long"]),
        "merch_lat": float(merch_lat),
        "merch_long": float(merch_long),
        "dob": str(base["dob"]),
        "merchant": base["merchant"],
        "city": base["city"],
        "state": base["state"],
        "job": base["job"],
        "zip": base["zip"],
        "Unnamed: 0": 0, "cc_num": 0, "first": "sim", "last": "sim",
        "street": "sim", "trans_num": "sim", "unix_time": 0,
    }


def _producer_loop(producer_started_at: str, conn_kwargs: dict):
    """
    Runs forever in a daemon thread, publishing one simulated transaction
    every PRODUCE_INTERVAL_SECONDS. Broad try/except per iteration so a
    transient broker hiccup logs and retries on the next tick instead of
    silently killing the thread (a dead daemon thread leaves no trace --
    Streamlit doesn't reraise exceptions from background threads).

    Connecting is itself inside the retry loop, not done once up front:
    against a REMOTE broker (Confluent Cloud) the very first connect can
    fail for reasons that clear on their own -- cold start, DNS, transient
    network -- and a connect exception raised outside the loop would kill
    this thread permanently, leaving a dashboard that renders fine and
    never receives a single row, with nothing in the UI to explain why.
    Reconnecting on demand also recovers from a broker that drops the
    connection later.

    conn_kwargs is passed in rather than computed here because
    _kafka_connection_kwargs() reads st.secrets, and Streamlit secrets
    access from a bare background thread is not something to rely on --
    the caller resolves it on the main thread and hands it over.
    """
    ref = load_reference_sample()
    producer = None

    while True:
        try:
            if producer is None:
                producer = KafkaProducer(
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    **conn_kwargs,
                )
                log.info("### IN-APP PRODUCER CONNECTED -> topic '%s' every %ds", KAFKA_TOPIC, PRODUCE_INTERVAL_SECONDS)

            row = generate_raw_transaction(ref)
            row["producer_started_at"] = producer_started_at
            producer.send(KAFKA_TOPIC, value=row)
            producer.flush()
            log.info("### PUBLISHED: category=%s amt=%.2f", row["category"], row["amt"])
        except Exception:
            log.exception("### PRODUCER CYCLE FAILED (will retry in %ds)", PRODUCE_INTERVAL_SECONDS)
            # Drop the client so the next iteration rebuilds it -- a
            # half-dead producer never recovers on its own.
            if producer is not None:
                try:
                    producer.close(timeout=1)
                except Exception:
                    pass
                producer = None
        time.sleep(PRODUCE_INTERVAL_SECONDS)


MODE_KAFKA = "kafka"
MODE_IN_PROCESS = "in-process"


@st.cache_resource(show_spinner="Checking for a Kafka broker...")
def get_pipeline_mode() -> str:
    """
    Decides once per process whether to run the Kafka streaming pipeline or
    the self-contained in-process fallback, by actually trying to reach the
    broker (which also validates SASL credentials, not just the hostname).

    Why a fallback exists at all: Kafka needs a broker, and there is no
    permanently-free hosted Kafka. The public Streamlit Community Cloud demo
    therefore has no broker to talk to, and without this the deployed app
    would render a dashboard that silently never receives a transaction.
    Rather than ship a dead demo or a paid dependency, the app degrades to
    generating transactions in-process -- the same simulation logic, just
    handed straight to the scorer instead of travelling through a broker.

    Everything downstream (feature engineering, hybrid scoring, SHAP,
    Postgres) is identical in both modes. The active mode is surfaced in the
    dashboard so it is never ambiguous which one is running.
    """
    try:
        from kafka import KafkaAdminClient
        admin = KafkaAdminClient(request_timeout_ms=5000, **_kafka_connection_kwargs())
        admin.close()
        log.info("### PIPELINE MODE: kafka (broker reachable)")
        return MODE_KAFKA
    except Exception as exc:
        log.warning("### PIPELINE MODE: in-process (no broker reachable: %s)", type(exc).__name__)
        return MODE_IN_PROCESS


@st.cache_resource
def start_producer_thread():
    """
    Starts the publishing loop exactly once per process (cached like
    start_scheduler() below, for the same reason: without caching, every
    Streamlit rerun would spawn a fresh thread instead of reusing one).
    daemon=True so it never blocks process shutdown.

    No-op in in-process mode: there is no broker to publish to, and
    run_inference() generates its own transactions instead.
    """
    if get_pipeline_mode() != MODE_KAFKA:
        log.info("### PRODUCER THREAD SKIPPED (in-process mode -- nothing to publish to)")
        return None

    started_at = datetime.now(timezone.utc).isoformat()
    # Resolve secrets here, on the main thread, and pass the plain dict in.
    conn_kwargs = _kafka_connection_kwargs()
    thread = threading.Thread(
        target=_producer_loop, args=(started_at, conn_kwargs), daemon=True, name="kafka-producer"
    )
    thread.start()
    log.info("### PRODUCER THREAD STARTED (fresh process, daemon=%s)", thread.daemon)
    return thread


def run_inference():
    """
    One pipeline cycle. In Kafka mode this drains whatever is currently on
    the topic; in in-process mode it generates a transaction directly.
    Either way the rows go through the identical scoring path below.
    """
    if get_pipeline_mode() != MODE_KAFKA:
        _run_inference_in_process()
        return
    _run_inference_kafka()


def _run_inference_in_process():
    """
    Fallback cycle: generate one transaction and score it immediately,
    with no broker involved. This is the pre-Kafka behaviour, kept alive so
    the app still does something useful wherever Kafka isn't available.
    """
    try:
        resources = load_models()
        row = generate_raw_transaction(load_reference_sample())
        row["producer_started_at"] = get_in_process_started_at()
        _record_stream_stat(row)
        _score_and_store(row, resources)
    except Exception:
        log.exception("### IN-PROCESS CYCLE FAILED")


@st.cache_resource
def get_in_process_started_at() -> str:
    """Fixed start time for in-process mode, so the dashboard's uptime
    metric means the same thing it does in Kafka mode."""
    return datetime.now(timezone.utc).isoformat()


def _run_inference_kafka():
    """
    Drains whatever new messages are currently sitting on the Kafka topic
    and scores each one. Data arrives over Kafka from either the in-app
    producer thread (start_producer_thread, started at bootstrap below) or
    a standalone `python producer.py` process -- both publish to the same
    topic, so this doesn't need to know which one is running.
    """
    try:
        resources = load_models()
        consumer = get_consumer()

        # poll() returns a dict of {TopicPartition: [records]}; drain everything
        # currently available rather than processing one message at a time.
        records = consumer.poll(timeout_ms=1000, max_records=20)
        total = sum(len(msgs) for msgs in records.values())
        log.info("### POLL RESULT: %d record(s) across %d partition(s)", total, len(records))
        if not records:
            return

        for tp, messages in records.items():
            for message in messages:
                _record_stream_stat(message.value)
                _score_and_store(message.value, resources)

    except Exception:
        log.exception("### INFERENCE CYCLE FAILED")


def _score_and_store(row: dict, resources: dict):
    """
    Runs one transaction dict (as received from Kafka) through the exact
    same feature engineering + hybrid inference + SHAP + Postgres write
    used in the original app.py. Unchanged from the original run_inference()
    body, aside from taking the raw row as a parameter instead of calling
    generate_raw_transaction() itself.
    """
    # producer_started_at is dashboard stream-health metadata (see
    # _record_stream_stat), not a model feature -- features.py's DROP_COLS
    # doesn't know about it, so left in it survives to engineer_features'
    # final df.astype("float64") and blows up on the ISO timestamp string.
    row = {k: v for k, v in row.items() if k != "producer_started_at"}
    raw_df = pd.DataFrame([row])
    amt = row["amt"]

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

    with _db_lock, get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO transactions
               (amt, category, xgb_score, ae_score, ae_threshold, final_flag, top_reason)
               VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (float(amt), str(raw_df["category"].iloc[0]), xgb_prob, ae_error,
             resources["ae_threshold"], final_flag, top_reason),
        )
        new_id = cur.fetchone()[0]
        conn.commit()
        log.info("### DB INSERT CONFIRMED: id=%d amt=%.2f flag=%d", new_id, amt, final_flag)


@st.cache_resource
def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_inference, "interval", seconds=POLL_INTERVAL_SECONDS, id="fraud_consumer_poll")
    scheduler.start()
    log.info(
        "### SCHEDULER STARTED (fresh process, id=%s): draining Kafka every %ds",
        id(run_inference), POLL_INTERVAL_SECONDS,
    )
    return scheduler


def fetch_recent(limit: int = 50) -> pd.DataFrame:
    from sqlalchemy import create_engine, text
    engine = create_engine(get_database_url())
    with engine.connect() as conn:
        return pd.read_sql(
            text("SELECT * FROM transactions ORDER BY id DESC LIMIT :limit"),
            conn, params={"limit": limit},
        )


# ---------------- App bootstrap ----------------
# NOTE: deliberately NOT calling get_consumer() here. Creating a
# KafkaConsumer connects eagerly and raises KafkaTimeoutError if the broker
# isn't reachable -- at module level that exception crashes the whole script
# and the dashboard renders as a stack trace instead of a page, with the
# producer thread below never even starting. That's a realistic first-boot
# state against a REMOTE broker (cold start, network blip, bad secret), so
# the consumer is left to run_inference(), which already creates it lazily
# inside a try/except and simply retries on the next 5s tick. The dashboard
# then still renders; consumer lag shows -1 until the broker answers.
init_db()
load_models()
start_scheduler()
start_producer_thread()

mode = get_pipeline_mode()

st.set_page_config(page_title="Hybrid Fraud Detection", layout="wide")
st.title("Hybrid Fraud Detection — Live Dashboard")
st.caption(
    "Supervised XGBoost + unsupervised Autoencoder, ensembled. "
    f"Scoring a new transaction every {POLL_INTERVAL_SECONDS}s."
)

# Which ingestion path is live is stated outright rather than left to be
# inferred -- the two modes produce an identical-looking dashboard, so
# showing this is the difference between disclosure and a misleading demo.
if mode == MODE_KAFKA:
    st.success(
        f"**Ingestion mode: Kafka streaming** — publishing to and consuming from topic "
        f"`{KAFKA_TOPIC}`, both inside this process.",
        icon="🔌",
    )
else:
    st.info(
        "**Ingestion mode: in-process simulation** — no Kafka broker reachable, so transactions "
        "are generated in-process and scored directly. The full Kafka streaming path is in this "
        "repo and runs locally via `docker compose up -d`; it needs a broker, and there is no "
        "permanently-free hosted Kafka, so the public demo runs this fallback instead.",
        icon="ℹ️",
    )

df = fetch_recent(50)
msgs_per_sec, producer_uptime = get_stream_metrics()

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Transactions (last 50)", len(df))
col2.metric("Flagged as fraud", int(df["final_flag"].sum()) if len(df) else 0)
col3.metric("Flag rate", f"{(100 * df['final_flag'].mean()):.1f}%" if len(df) else "—")
# Consumer lag is a Kafka concept -- meaningless without a broker, so it's
# shown as "n/a" rather than a misleading 0 or -1.
col4.metric("Kafka consumer lag", get_consumer_lag() if mode == MODE_KAFKA else "n/a")
col5.metric("Transactions/sec", f"{msgs_per_sec:.2f}")
col6.metric("Pipeline uptime", producer_uptime or "—")

st.subheader("Recent transactions")
st.dataframe(df, width="stretch")

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
    st.info("Waiting for the first transactions to be scored...")

st.button("Refresh now")