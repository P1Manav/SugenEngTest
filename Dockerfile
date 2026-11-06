FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install minimal build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first to keep image lightweight and avoid multi-GB CUDA overhead
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application source code
COPY scripts/ scripts/
COPY text_utils.py .
COPY assertion_filter.py .
COPY diagnosis_extractor.py .
COPY icd_hierarchy.py .
COPY retrieval.py .
COPY reranker.py .
COPY pipeline.py .
COPY build_kb.py .
COPY build_index.py .
COPY entrypoint.py .
COPY evaluate.py .

# Copy MMS source data
COPY data/mms-2026-01.json data/mms-2026-01.json

# Copy prebuilt artifacts or build them at build time
COPY models/ models/
COPY kb/ kb/

# Ensure models are cached and indexes are built at build time
RUN python scripts/download_models.py
RUN python build_kb.py data/mms-2026-01.json
RUN python build_index.py kb/enriched_kb.json models/all-MiniLM-L6-v2

ENTRYPOINT ["python", "entrypoint.py"]
