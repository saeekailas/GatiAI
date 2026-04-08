FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files
COPY pyproject.toml . 
COPY requirements.txt .

# Install dependencies using uv
RUN uv pip install --system --no-cache -r requirements.txt

# Copy source code
COPY . .

# Hugging Face port
EXPOSE 7860

# Start server using uv
CMD ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]