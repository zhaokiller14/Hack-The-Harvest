# ── Build stage: install dependencies ────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# System libs needed by rasterio, opencv, shapely, torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgdal-dev libgeos-dev libproj-dev \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Strip dev-only packages before installing
RUN sed '/ipykernel/d; /jupyter/d; /seaborn/d' requirements.txt > requirements.prod.txt && \
    pip install --no-cache-dir --prefix=/install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.prod.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

# Runtime system libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev libgeos-dev libproj-dev \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# App source
COPY app/       ./app/
COPY frontend/  ./frontend/
COPY models/    ./models/

# Non-root user for security
RUN useradd -m -u 1000 ardhi && chown -R ardhi:ardhi /app
USER ardhi

EXPOSE 8000

# GEE credentials are mounted at runtime via volume (see docker-compose.yml)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
