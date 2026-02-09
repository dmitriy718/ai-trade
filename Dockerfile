# ============================================
# AI Trading Bot - Production Dockerfile
# Multi-stage build, ARM64 optimized
# ============================================

# Stage 1: Builder
FROM python:3.11-slim as builder

# Set build arguments
ARG TARGETARCH=arm64

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim as runtime

# Security: Run as non-root
RUN groupadd -r trader && useradd -r -g trader -d /app -s /sbin/nologin trader

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY static/ ./static/
COPY config/ ./config/
COPY main.py .
COPY requirements.txt .

# Create required directories
RUN mkdir -p data logs models && \
    chown -R trader:trader /app

# Switch to non-root user
USER trader

# Environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    TRADING_MODE=paper \
    LOG_LEVEL=INFO \
    DASHBOARD_HOST=0.0.0.0 \
    DASHBOARD_PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${DASHBOARD_PORT}/api/v1/status || exit 1

# Expose dashboard port
EXPOSE ${DASHBOARD_PORT}

# Use tini for proper signal handling
ENTRYPOINT ["tini", "--"]

# Run the bot
CMD ["python", "main.py"]
