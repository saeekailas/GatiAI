FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .
# FIX: Ensure we have the latest pip to avoid the version errors you saw earlier
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . . 

# HuggingFace Spaces uses port 7860
ENV PORT=7860
EXPOSE 7860

# Simplified CMD: Direct execution is more reliable in Docker than bash strings
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]