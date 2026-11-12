FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

# Install minimal build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU pinned to keep image lightweight
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Copy and install pinned python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application source code
COPY scripts/ scripts/
COPY text_utils.py .
COPY assertion_filter.py .
COPY diagnosis_extractor.py .
COPY clinical_concepts.py .
COPY icd_hierarchy.py .
COPY retrieval.py .
COPY reranker.py .
COPY multilingual.py .
COPY pipeline.py .
COPY pipeline_v2.py .
COPY llm_reranker.py .
COPY build_kb.py .
COPY build_index.py .
COPY entrypoint.py .
COPY evaluate.py .
COPY score.py .
COPY rules/ rules/
COPY tests/ tests/

# Copy MMS source data and dev data
COPY data/ data/

# Download and verify pinned model weights at build time
RUN python scripts/download_models.py

# Copy prebuilt knowledge base artifacts
COPY kb/ kb/

ENTRYPOINT ["python", "entrypoint.py"]
