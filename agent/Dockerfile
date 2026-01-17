FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install pandas numpy==1.24.3 scikit-learn==1.3.0 requests psycopg2-binary joblib

# Copy Model Artifacts
COPY research/model.joblib /app/model.joblib
COPY research/model_metadata.json /app/model_metadata.json

# Copy Historical Data for fallback
COPY research/data/btc_usdt_1h.csv /app/data/btc_usdt_1h.csv

# Copy Agent Code
COPY agent/run.py /app/run.py

# Env vars
ENV MODEL_PATH=/app/model.joblib
ENV META_PATH=/app/model_metadata.json
ENV DB_HOST=postgres
ENV DB_PORT=5432
ENV DB_USER=postgres
ENV DB_PASSWORD=password
ENV DB_NAME=stocky

# Run
CMD ["python", "-u", "run.py"]
