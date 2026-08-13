"""
producer.py
Standalone process that publishes simulated "live" transactions to the
Kafka topic 'transactions-stream'. This replaces the in-process
generate_raw_transaction() call that used to live inside app.py's
run_inference(). The generation LOGIC is unchanged — only where it runs
and how the result is handed off (Kafka instead of a direct function
return) is different.

Run in its own terminal, alongside `streamlit run app.py`:
    python producer.py
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
from kafka import KafkaProducer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "transactions-stream"
PUBLISH_INTERVAL_SECONDS = 5  # how often a new "live" transaction is emitted


def load_reference_sample() -> pd.DataFrame:
    return joblib.load("reference_sample.pkl")


def generate_raw_transaction(ref: pd.DataFrame) -> dict:
    """
    Identical logic to the original generate_raw_transaction() in app.py —
    resamples a real legitimate transaction and, ~8% of the time, perturbs
    it into a believable outlier. The only change: returns a plain
    JSON-serializable dict (datetimes as ISO strings) instead of a
    pandas DataFrame, since this now travels over Kafka.
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

    row = {
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
    return row


def main():
    ref = load_reference_sample()
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    # Stamped once and carried on every message so the dashboard can derive
    # "producer uptime" from the most recently consumed message alone,
    # without needing a separate heartbeat channel.
    producer_started_at = datetime.now(timezone.utc).isoformat()
    log.info("Producer started -> topic '%s' every %ds", KAFKA_TOPIC, PUBLISH_INTERVAL_SECONDS)

    while True:
        row = generate_raw_transaction(ref)
        row["producer_started_at"] = producer_started_at
        producer.send(KAFKA_TOPIC, value=row)
        producer.flush()
        log.info("Published: category=%s amt=%.2f", row["category"], row["amt"])
        time.sleep(PUBLISH_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()