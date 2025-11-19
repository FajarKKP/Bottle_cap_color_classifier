# ---- Base image ----
FROM python:3.11-slim AS base

# Set working directory
WORKDIR /app

# ---- Install system dependencies (minimal) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Poetry ----
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir poetry

# ---- Copy dependency files first (for caching) ----
COPY pyproject.toml poetry.lock ./

# ---- Install dependencies WITHOUT creating a virtual environment ----
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# ---- Copy the rest of the application ----
COPY . .

# ---- Set default entrypoint ----
ENTRYPOINT ["python", "-m", "bsort.cli"]
