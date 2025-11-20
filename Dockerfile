# ---- Stage 1: Build runtime dependencies only ----
FROM python:3.11-slim AS builder

WORKDIR /app

# Minimal system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir poetry

# Copy only dependency files first for caching
COPY pyproject.toml poetry.lock ./

# Disable virtualenv creation and install ONLY runtime dependencies (exclude dev)
RUN poetry config virtualenvs.create false \
    && poetry export -f requirements.txt --without-hashes --without-urls --without-credentials --only main > requirements_runtime.txt \
    && pip install --no-cache-dir -r requirements_runtime.txt

# ---- Stage 2: Final runtime image ----
FROM python:3.11-slim AS runtime

WORKDIR /app

# Minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project files
COPY . .

# Install CPU-only PyTorch separately
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu

# Set default entrypoint
ENTRYPOINT ["python", "-m", "bsort.cli"]
