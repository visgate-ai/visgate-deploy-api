# Multi-stage build: builder + runtime, target <500MB
# Python 3.11, non-root user, healthcheck, graceful shutdown

# ---- Builder ----
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---- Runtime ----
FROM python:3.11-slim AS runtime

WORKDIR /app

# Non-root user
RUN groupadd --gid 1000 app && useradd --uid 1000 --gid app --shell /bin/false app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/app/.local
ENV PATH=/home/app/.local/bin:$PATH

# Application
COPY src/ ./src/
COPY requirements.txt .

# Cloud Run uses PORT env
ENV PORT=8080
EXPOSE 8080

USER app

# Graceful shutdown: uvicorn handles SIGTERM
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=2 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health')" || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
