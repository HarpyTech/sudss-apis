# Use PyTorch base image (latest available, CUDA runtime)
FROM pytorch/pytorch:2.1.2-cpu


WORKDIR /app

# Copy requirements first and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose FastAPI port
EXPOSE 8080

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
