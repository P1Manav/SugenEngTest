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

# Install PyTorch CPU pinned to keep image lightweight and avoid multi-GB CUDA overhead
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Copy and install pinned python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application source code and tests
COPY scripts/ scripts/
COPY text_utils.py .
COPY assertion_filter.py .
COPY diagnosis_extractor.py .
COPY icd_hierarchy.py .
COPY retrieval.py .
COPY reranker.py .
COPY llm_reranker.py .
COPY multilingual.py .
COPY pipeline.py .
COPY build_kb.py .
COPY build_index.py .
COPY entrypoint.py .
COPY evaluate.py .
COPY score.py .
COPY tests/ tests/

# Copy MMS source data and dev data
COPY data/ data/

# Copy prebuilt artifacts and vendored model weights (no external downloads at build or run time)
COPY models/ models/
COPY kb/ kb/

ENTRYPOINT ["python", "entrypoint.py"]
