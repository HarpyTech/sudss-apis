    # Multi-stage Dockerfile for Med-Gemma (Service B)
    FROM python:3.11-slim AS builder
    WORKDIR /app

    RUN apt-get update \
&& apt-get install -y --no-install-recommends build-essential gcc libffi-dev \
&& rm -rf /var/lib/apt/lists/*

    COPY common /app/common
    COPY med-gemma-api /app/app

    RUN python -m pip install --upgrade pip setuptools wheel \
&& pip install /app/common \
&& cd /app/app \
&& pip install -r requirements.txt

    FROM python:3.11-slim
    WORKDIR /app
    COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
    COPY --from=builder /usr/local/bin /usr/local/bin

    COPY common /app/common
    COPY med-gemma-api /app/app

    EXPOSE 8080
    WORKDIR /app/app
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
