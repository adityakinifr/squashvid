FROM python:3.11-slim

# Cache bust: v8 - force full rebuild
ARG CACHEBUST=8

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
ARG PIP_BUST=2
RUN pip install --no-cache-dir -r requirements.txt && pip list | grep -i supabase

# Copy application code
COPY squashvid/ ./squashvid/
COPY start.py .

# Set environment
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run the app - Railway provides PORT env var
CMD python start.py
