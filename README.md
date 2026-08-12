# Fraud LLM

Local fraud detection demo using Streamlit, XGBoost, and an autoencoder ensemble.

## Run locally

1. Create a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run:
   ```bash
   streamlit run app.py
   ```

## Hugging Face Spaces

This repo is a Streamlit app, not a static site. If you want to deploy to Hugging Face Spaces, use the `Docker` SDK or convert the app into a static frontend.

## Notes

- `transactions.db` is excluded from version control.
- Model artifact files are included in the repo for local deployment.
- If model files get larger, consider using Git LFS.
