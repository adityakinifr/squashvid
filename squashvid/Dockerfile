FROM python:3.11-slim

# Cache bust: v3
ARG CACHEBUST=3

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . ./squashvid/

# Set environment
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run the app - Railway provides PORT env var
CMD uvicorn squashvid.api:app --host 0.0.0.0 --port ${PORT:-8000}
