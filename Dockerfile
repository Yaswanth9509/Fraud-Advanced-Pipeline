FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (separate layer) so Docker caches this step
# and doesn't reinstall everything every time app.py changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project (code + trained model files)
COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
