    # Multi-stage Dockerfile for Gemini Pro (Service A)
    FROM python:3.11-slim AS builder
    WORKDIR /app

    # Optional: install build deps for packages that need compilation
    RUN apt-get update \
&& apt-get install -y --no-install-recommends build-essential gcc libffi-dev \
&& rm -rf /var/lib/apt/lists/*

    # Copy the shared common library and the service source into the image (context is repo root)
    COPY common /app/common
    COPY gemini-pro-api /app/app

    # Install the common package first (as required) and then install the service requirements
    RUN python -m pip install --upgrade pip setuptools wheel \
&& pip install /app/common \
&& cd /app/app \
&& pip install -r requirements.txt

    # Final runtime image
    FROM python:3.11-slim
    WORKDIR /app

    # Copy installed Python packages from builder
    COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
    COPY --from=builder /usr/local/bin /usr/local/bin

    # Copy app code + common for clarity (not strictly necessary if dependencies already installed)
    COPY common /app/common
    COPY gemini-pro-api /app/app

    EXPOSE 8080
    WORKDIR /app/app
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
