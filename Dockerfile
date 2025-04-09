FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly install uvicorn
RUN pip install --no-cache-dir uvicorn fastapi

# Copy source code and create necessary directories
COPY . .
RUN mkdir -p data

# Set environment variables
ENV PYTHONPATH=/app

# Run the application
CMD ["python", "src/main.py"]