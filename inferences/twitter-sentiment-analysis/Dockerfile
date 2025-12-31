FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app:/app/common:$PYTHONPATH

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    git build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy project metadata
COPY app.py .
COPY pyproject.toml .
COPY uv.lock .


# Install dependencies using uv, then export and install with pip to system
RUN uv sync --frozen --no-dev && \
    uv pip install -e . --system

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
