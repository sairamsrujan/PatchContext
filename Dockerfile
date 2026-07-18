# PatchContext — Docker image (Cloud Run / HF Docker Space / local).
# Local check: docker build -t patchcontext . && docker run -p 8501:8501 --env-file .env patchcontext
#
# Layer order is deliberate for fast rebuilds: the slow layers (pip install,
# model pre-download) depend only on requirements.txt and fixed model IDs, so a
# code change rebuilds only the cheap final layers instead of re-downloading
# ~2.4 GB of model weights every time.

FROM python:3.11-slim

# HF Spaces / Cloud Run run containers as UID 1000; its default HOME isn't writable.
RUN useradd -m -u 1000 appuser

WORKDIR /app

# HF_HOME points model caches at a writable dir. MODEL_DEVICE=cpu (no GPU).
# FAISS is no longer imported in the serving process (retriever.py uses NumPy),
# so torch is the only OpenMP user — the x86-64 torch+faiss teardown segfault
# is gone and torch may use both vCPUs. NumPy's BLAS stays single-threaded since
# retrieval is a tiny 12k×1024 matmul.
ENV HF_HOME=/app/.cache \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DEVICE=cpu \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# CPU-only torch (the default PyPI wheel bundles ~2 GB of CUDA). Cached unless
# requirements.txt changes.
COPY requirements.txt ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Pre-download the three models BEFORE copying app code, as appuser into its own
# cache. This layer depends only on the (fixed) model IDs, so it stays cached
# across code changes — the key to fast rebuilds — and avoids cold-start weight
# fetches at runtime.
RUN mkdir -p /app/.cache && chown -R appuser:appuser /app
USER appuser
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('Qwen/Qwen3-Embedding-0.6B'); \
CrossEncoder('BAAI/bge-reranker-base'); \
CrossEncoder('cross-encoder/nli-deberta-v3-base')"

# App code + editable install last: a code change rebuilds only from here (~1-2
# min). Code is root-owned but world-readable; appuser runs it and only writes
# to its own /app/.cache.
USER root
COPY . .
RUN pip install -e .
USER appuser

EXPOSE 8501
# fileWatcherType none: streamlit's watcher walks transformers' lazy imports and
# crashes on optional submodules (torchvision). PORT: Cloud Run injects $PORT
# (8080); local/HF default to 8501.
CMD streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.fileWatcherType=none
