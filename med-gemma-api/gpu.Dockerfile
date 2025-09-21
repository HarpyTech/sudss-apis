# syntax=docker/dockerfile:1

################################################################################
# Production GPU Dockerfile for MedGemma FastAPI
#
# - Base: NVIDIA CUDA runtime (CUDA 12.8 / cu128). Change tag if you need another CUDA.
# - Installs PyTorch wheel for the matching CUDA via TORCH_INDEX_URL build-arg.
# - Installs remaining Python deps from requirements.txt.
# - Uses gunicorn + uvicorn worker for production.
# - Runs as non-root user.
################################################################################

ARG BASE_IMAGE="nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04"
FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
ARG PIP_EXTRA_INDEX_URL=""
ARG WORKERS=2

ENV PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:${PATH}" \
    MEDGEMMA_VARIANT="4b-it" \
    USE_QUANTIZATION="true" \
    DEFAULT_PROMPT="Describe this X-ray" \
    LOG_LEVEL="INFO" \
    UVICORN_LIMIT_CONCURRENCY=0 \
    UVICORN_LIMIT_MAX_REQUESTS=0

WORKDIR /app

# -----------------------
# System packages needed for runtime
# -----------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    python3 \
    python3-distutils \
    python3-venv \
    python3-pip \
    libjpeg-dev \
    libpng-dev \
    libsndfile1 \
    libgomp1 \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy requirements
COPY ./med-gemma-api/req-prod.txt /app/requirements.txt

# -----------------------
# Install PyTorch (GPU) first to leverage caching
# -----------------------
# Pass TORCH_INDEX_URL as build-arg to select appropriate CUDA wheel.
RUN python3 -m pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} \
    "torch" "torchvision" "torchaudio" || (echo "PyTorch install failed; check TORCH_INDEX_URL and base image" && exit 1)

# -----------------------
# Install remaining Python dependencies
# -----------------------
RUN if [ -n "${PIP_EXTRA_INDEX_URL}" ]; then \
    python3 -m pip install --no-cache-dir --extra-index-url ${PIP_EXTRA_INDEX_URL} -r /app/requirements.txt; \
    else \
    python3 -m pip install --no-cache-dir -r /app/requirements.txt; \
    fi

# -----------------------
# Copy application code & test images (if present)
# -----------------------
COPY ./med-gemma-api /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

# Expose port
EXPOSE 8000

# Healthcheck (simple): returns 200 when uvicorn responds
# Adjust interval and retries in production as needed
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# -----------------------
# Run: Gunicorn with Uvicorn worker
# - WORKERS build-arg can be overridden at docker run with env var.
# - Use --preload if you want to load the model in master process (reduces mem duplication with worker for some deployments).
# -----------------------
ENV GUNICORN_CMD_ARGS="--bind=0.0.0.0:8000 --timeout 120"

# Use a shell form that allows env var substitution for WORKERS; you can override with docker run -e WORKERS=4
CMD exec gunicorn -k uvicorn.workers.UvicornWorker app:app \
    --workers ${WORKERS} \
    ${GUNICORN_CMD_ARGS}
