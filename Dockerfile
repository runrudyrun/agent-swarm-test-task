# Optimized Dockerfile for faster builds
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    TRANSFORMERS_CACHE=/home/appuser/.cache/infinitepay \
    HF_HOME=/home/appuser/.cache/infinitepay \
    HOME=/home/appuser

# Install system dependencies needed for ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with home directory
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

# Set working directory
WORKDIR /app

# Copy and install Python dependencies first (for better caching)
COPY pyproject.toml ./
RUN pip install --upgrade pip setuptools wheel && pip install .

# Copy application code
COPY --chown=appuser:appuser . .

# Copy entrypoint scripts
COPY --chown=appuser:appuser docker-entrypoint.sh /app/docker-entrypoint.sh
COPY --chown=appuser:appuser docker_startup_check.py /app/docker_startup_check.py

# Create necessary directories with proper ownership
RUN mkdir -p /app/data/{raw,chroma,mock,sources} && \
    mkdir -p /home/appuser/.cache/infinitepay && \
    chmod +x /app/docker-entrypoint.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use the entrypoint script
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD []