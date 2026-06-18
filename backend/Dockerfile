FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps commonly needed by sentence-transformers/torch

RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# Install uv

RUN pip install --no-cache-dir uv

# Copy dependency manifests first for better layer caching

COPY pyproject.toml uv.lock ./

# Create venv and install dependencies

RUN uv sync --frozen

# Copy application source

COPY . .

CMD sh -c '.venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port ${PORT:-10000} --log-level debug'
