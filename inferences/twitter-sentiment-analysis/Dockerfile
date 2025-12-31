# --- Base image ---
FROM python:3.10-slim

# --- Environment variables ---
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache
ENV HF_DATASETS_CACHE=/app/model_cache
ENV HF_METRICS_CACHE=/app/model_cache
ENV VENV_PATH=/app/.venv

# --- Set working directory ---
WORKDIR /app

# --- Install system dependencies ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        python3-venv \
        build-essential \
        curl \
        && rm -rf /var/lib/apt/lists/*

# --- Upgrade pip and install uv ---
RUN pip install --upgrade pip setuptools wheel
RUN pip install uv

# --- Copy project files including pyproject.toml ---
COPY . /app

# --- Create virtual environment named .venv and sync dependencies ---
RUN uv venv create $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"
RUN uv sync

# --- Pre-download Hugging Face model for faster startup ---
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('mishrabp/bert-base-uncased-tweet-sentiment-analysis'); \
AutoModelForSequenceClassification.from_pretrained('mishrabp/bert-base-uncased-tweet-sentiment-analysis')"

# --- Expose Hugging Face Space port ---
EXPOSE 7860

# --- Run Streamlit app using the .venv environment ---
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
