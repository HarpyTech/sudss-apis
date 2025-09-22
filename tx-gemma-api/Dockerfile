FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Torch CPU first (avoid pulling CUDA wheels)
RUN pip install -U pip \
 && pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# Bring in shared code and service
COPY common ./common
COPY tx-gemma-api ./tx-gemma-api

# Install shared + service deps
RUN pip install -e ./common \
 && pip install -r tx-gemma-api/requirements.txt

WORKDIR /app/tx-gemma-api
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
