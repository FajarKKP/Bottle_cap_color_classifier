# ---- Stage 1: Build runtime dependencies only ----
FROM python:3.11-slim AS builder

WORKDIR /app

# Minimal system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry (stable) and pip
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "poetry>=1.5.1"

# Copy dependency files for caching
COPY pyproject.toml poetry.lock ./

# Disable virtualenvs and install ONLY runtime dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi \
    && rm -rf /root/.cache/pypoetry/* /root/.cache/pip/*

# ---- Stage 2: Final runtime image ----
FROM python:3.11-slim AS runtime

WORKDIR /app

# Minimal system dependencies for runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project files
COPY . .

# Install CPU-only PyTorch separately (if needed)
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu

# Set default entrypoint
ENTRYPOINT ["python", "-m", "bsort.cli"]
