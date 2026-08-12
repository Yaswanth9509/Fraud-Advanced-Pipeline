"""
generate_reference_sample.py
Samples real legitimate transactions from fraudTrain.csv so the live
simulator (app.py) can draw statistically realistic feature values instead
of pure random noise, which was inflating the false-flag rate.

Run once, locally:  python generate_reference_sample.py
Produces: reference_sample.pkl  (place alongside app.py before deploying)

This does NOT retrain the models — it's a fast, separate step.
"""

import logging

import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

TRAIN_PATH = "data/fraudTrain.csv"
SAMPLE_SIZE = 5000
KEEP_COLS = [
    "category", "amt", "gender", "lat", "long", "merch_lat", "merch_long",
    "dob", "merchant", "city", "state", "job", "zip",
]


def main():
    log.info("Loading %s ...", TRAIN_PATH)
    df = pd.read_csv(TRAIN_PATH)

    legit = df[df["is_fraud"] == 0]
    n = min(SAMPLE_SIZE, len(legit))
    sample = legit.sample(n=n, random_state=42).reset_index(drop=True)[KEEP_COLS]

    joblib.dump(sample, "reference_sample.pkl")
    log.info("Saved %d reference legitimate transactions to reference_sample.pkl", n)


if __name__ == "__main__":
    main()
