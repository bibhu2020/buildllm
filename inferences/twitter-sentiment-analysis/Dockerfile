# --- Base image ---
FROM python:3.10-slim

# --- Environment variables ---
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Hugging Face cache (persist across restarts)
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache
ENV HF_DATASETS_CACHE=/app/model_cache
ENV HF_METRICS_CACHE=/app/model_cache

# Virtual environment path
ENV VENV_PATH=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# --- Set working directory ---
WORKDIR /app

# --- Install system dependencies ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- Install uv (official installer, NOT pip) ---
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# --- Copy project files ---
COPY . /app

# --- Create .venv and install dependencies from pyproject.toml ---
RUN uv sync --venv /app/.venv

# --- Pre-download model to speed up startup ---
RUN python - <<EOF
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "mishrabp/bert-base-uncased-tweet-sentiment-analysis"
AutoTokenizer.from_pretrained(model_id)
AutoModelForSequenceClassification.from_pretrained(model_id)
EOF

# --- Expose Hugging Face Space port ---
EXPOSE 7860

# --- Run Streamlit app ---
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
