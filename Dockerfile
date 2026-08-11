# Dockerfile

# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Dependencies in one layer so the compilers can be purged again. build-essential is
# only needed to build wheels; in its own layer it would stay in the image forever.
# curl remains for the healthcheck, libgomp and openblas are runtime deps of faiss.
#
# pip>=25.1 resumes interrupted downloads; the base image ships 24.0, which restarts
# multi-hundred-MB CUDA wheels from zero on any mid-stream TLS break.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl libgomp1 libopenblas-dev \
    && pip install --upgrade "pip>=25.1" \
    && pip install --retries 10 --timeout 120 --resume-retries 10 -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create user and cache directory
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/.cache/huggingface \
    && chown -R appuser:appuser /app/.cache

# Model weights are ~4GB. This layer is keyed only on the model names and preload.py,
# so ordinary application changes never invalidate it. ARGs stay build-scoped and do
# not leak into the runtime environment, where .env remains the source of truth.
ARG EMBEDDING_MODEL=BAAI/bge-m3
ARG RERANKER_MODEL=BAAI/bge-reranker-v2-m3

COPY app/preload.py ./app/

# Switch to appuser for model download (Security + Ownership)
USER appuser

RUN python app/preload.py

# Final app copy
COPY --chown=appuser:appuser app/ ./app/

# A named volume inherits ownership from the image path it is seeded from, so this
# directory must exist and belong to appuser or the mounted volume arrives root-owned
# and unwritable. /app itself is root-owned, hence the brief elevation. Kept after the
# model layer so application changes never invalidate the 4GB download.
USER root
RUN mkdir -p /app/data/index && chown -R appuser:appuser /app/data
USER appuser

# =========================
# Targets
# =========================
FROM base AS backend
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS frontend
EXPOSE 8501
CMD ["streamlit", "run", "app/frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]