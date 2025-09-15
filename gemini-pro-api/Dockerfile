# Multi-stage Dockerfile for Gemini Pro (Service A)
FROM python:3.9-slim

WORKDIR /app

# Copy only the service files
COPY gemini-pro-api/requirements.txt .
COPY gemini-pro-api/main.py .

# Install dependencies with pip upgrade
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
