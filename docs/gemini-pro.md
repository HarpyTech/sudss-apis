# Gemini Pro API Service Setup Guide

## Prerequisites
- Docker installed on your system
- Google AI API key

## Build and Run Docker Container

Navigate to the project root directory and run these commands:

```bash
# Build the Docker image
docker build -t gemini-pro-api:latest -f gemini-pro-api/Dockerfile .

# Run the container (replace YOUR_API_KEY with your actual Google AI API key)
docker run -p 8080:8080 -e GOOGLE_API_KEY=YOUR_API_KEY gemini-pro-api:latest
```

## Testing the API

### Text Generation Endpoint
Test the API using curl:

```bash
curl -X POST "http://localhost:8080/generate" ^
-H "Content-Type: application/json" ^
-d "{\"prompt\":\"Tell me a joke\",\"model\":\"gemini-2.5-pro\"}"
```

### Expected Response Format
```json
{
    "model": "gemini-2.5-pro",
    "prompt": "Tell me a joke",
    "response": "Generated response from Gemini",
    "timestamp": "2024-01-01T12:00:00.000Z"
}
```

## Supported Models
- gemini-2.5-flash
- gemini-2.5-pro

## Important Notes
- The API key is passed as an environment variable for security
- The service runs on port 8080
- All timestamps are in UTC
- For local development, you can use the FastAPI swagger UI at `http://localhost:8080/docs`

## Troubleshooting
If you encounter any issues:
1. Verify your API key is correct
2. Check Docker logs: `docker logs <container_id>`
3. Ensure port 8080 is not in use by another service