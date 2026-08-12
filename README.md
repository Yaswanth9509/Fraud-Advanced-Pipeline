# 🛡️ Fraud-Advanced-Pipeline: Real-Time Fraud & Anomaly Detection System

A hybrid fraud-detection system combining **supervised machine learning** and **unsupervised deep learning** to monitor, flag, and analyze transaction fraud in real time. It features a live Streamlit dashboard, a database audit trail, and explanation reasoning using SHAP values.

---

## 📈 Dashboard Preview

![Streamlit Real-Time Dashboard](streamlit_realtime.png)

---

## 🤖 Why These Algorithms Are Used

This pipeline utilizes a **hybrid ensemble architecture** combining two distinct models to build a defense-in-depth security layer:

### 1. Supervised Learning: XGBoost
* **Why**: XGBoost (Extreme Gradient Boosting) is the gold standard for tabular data classification. It builds decision trees sequentially to minimize a loss function.
* **Role**: It excels at recognizing **known signatures of fraud** that exist in historical training data (e.g., specific transaction categories paired with high values or specific hours).
* **Training Detail**: The training data features a severe class imbalance (<1% fraud transactions). During training, **SMOTE** (Synthetic Minority Over-sampling Technique) is applied to balance the classes and allow the model to learn fraud boundaries effectively.

### 2. Unsupervised Learning: Deep Autoencoder
* **Why**: Fraud patterns constantly evolve. A supervised model alone cannot flag a fraud technique it has never seen before.
* **Role**: The Autoencoder (a Neural Network implemented in TensorFlow/Keras) acts as an **anomaly detector** to flag **unseen/novel fraud patterns**.
* **Training Detail**: The network is trained *only* on legitimate (non-fraudulent) transactions. It learns to compress transaction features into a low-dimensional bottleneck and reconstruct them.
  * When a **normal** transaction is input, the model reconstructs it with high accuracy (low Reconstruction Error).
  * When an **anomalous** transaction is input (e.g., highly unusual amount, distant location, odd hour), the model fails to reconstruct it accurately (high Reconstruction Error).
  * If the reconstruction error (MSE) exceeds a pre-calibrated threshold, the transaction is flagged as an anomaly.

### 3. Hybrid Decision Logic
The final flag is triggered if **either** model flags the transaction:
$$\text{Final Flag} = \text{XGBoost Flag (Probability } > 0.5) \lor \text{Autoencoder Anomaly (Error } > \text{Threshold)}$$
This captures both highly typical fraud profiles and completely new types of anomalies.

---

## 📁 File Structure

```text
Fraud LLM/
├── data/
│   ├── fraudTrain.csv            # [LOCAL ONLY] Kaggle Training Set (~351 MB)
│   └── fraudTest.csv             # [LOCAL ONLY] Kaggle Test Set (~150 MB)
├── .gitignore                    # Prevents large data files from being tracked
├── Dockerfile                    # Containerization config for deployment
├── README.md                     # Project documentation (this file)
├── requirements.txt              **# Project library dependencies**
├── features.py                   **# Shared feature engineering pipeline (avoids train/serve skew)**
├── train_model.py                **# Model training, threshold calibration & artifact serialization**
├── generate_reference_sample.py  **# Utility to build lightweight simulation reference sample**
├── app.py                        **# Streamlit web app, simulation scheduler, and inference logic**
├── transactions.db               # Local SQLite database containing simulated audit trail
├── streamlit_realtime.png        # Real-time screenshot of running Streamlit app
├── xgb_model.pkl                 # [Generated] Trained XGBoost classifier
├── autoencoder_model.keras       # [Generated] Trained Keras Autoencoder model
├── scaler.pkl                    # [Generated] StandardScaler for normalizing numeric features
├── ae_threshold.pkl              # [Generated] Calibrated anomaly threshold (Float)
├── feature_columns.pkl           # [Generated] List of feature columns for schema alignment
├── category_maps.pkl             # [Generated] Frequency maps for high-cardinality columns
└── reference_sample.pkl          # [Generated] 1,000 genuine transactions for dashboard simulation
```

---

## ⚙️ Generated Files & Their Purpose

To run the Streamlit app (`app.py`), the model training script (`train_model.py`) generates several files. These files bridge the training phase to the real-time inference phase:

| Generated File | Type | Why It Is Generated & What It Does |
| :--- | :--- | :--- |
| **`xgb_model.pkl`** | Joblib Serialized Model | Stores the trained XGBoost model parameters so we can run instant predictions in `app.py` without retraining. |
| **`autoencoder_model.keras`** | TensorFlow H5/Keras Model | Stores the neural network weights and structure of the Autoencoder. Loaded by TensorFlow in `app.py` to calculate reconstruction errors. |
| **`scaler.pkl`** | Joblib Serialized Scaler | Contains the mean and variance computed by `StandardScaler` during training. This ensures live inference data is scaled **exactly** like the training data. |
| **`ae_threshold.pkl`** | Joblib Serialized Float | Stores the anomaly boundary (the 97.5th percentile of validation reconstruction errors). Any live transaction with an error exceeding this is flagged. |
| **`feature_columns.pkl`** | Joblib Serialized List | Lists the exact column names and order of the model's feature set. Used to align incoming data formats and prevent schema errors. |
| **`category_maps.pkl`** | Joblib Serialized Dictionary | Contains frequency mapping percentages for high-cardinality fields (like ZIP codes or Jobs) so they can be represented numerically at inference time. |
| **`reference_sample.pkl`** | Joblib Serialized DataFrame | Contains a subset of 1,000 real, legitimate transactions. The simulator in `app.py` randomly samples from this to create realistic live transactions (with real correlation profiles) rather than generating noisy random synthetic values. |
| **`transactions.db`** | SQLite Database | A local database automatically created by `app.py` to store simulated transactions, flags, and model scores as a persistent audit history. |

---

## 🚀 Usage & Deployment

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Model Training (Optional)
If you wish to retrain the models:
1. Download the Kaggle **Fraud Detection Dataset** (kartik2112/fraud-detection).
2. Create a folder named `data/` and place `fraudTrain.csv` and `fraudTest.csv` inside it.
3. Run the training script:
   ```bash
   python train_model.py
   ```
This will fit the models and regenerate the `.pkl` and `.keras` files.

### 3. Generate Reference Sample
If you need to regenerate the reference sample used by the simulator from the dataset:
```bash
python generate_reference_sample.py
```

### 4. Running the Dashboard
Start the live dashboard and real-time transaction simulator:
```bash
streamlit run app.py
```

---

## 💼 Real-World Application & Value

This hybrid architecture represents a production-grade approach to fraud risk management:

* **Train/Serve Skew Prevention**: By importing `features.py` inside both `train_model.py` and `app.py`, the exact same feature extraction transformations (distance calculation, frequency encoding, dummy categorization) are applied, eliminating a common cause of model decay in production.
* **Explainable AI (XAI)**: Includes **SHAP (SHapley Additive exPlanations)** values integrated directly into the dashboard alerts, showing investigators the exact top contributors (e.g., amount, location distance, category) that led XGBoost to flag a transaction.
* **Audit Trail**: Every transaction is logged into a structured SQLite database. Real-world financial institutions rely on these database logs for regulatory compliance (AML/CFT auditing) and retrospective analysis.
