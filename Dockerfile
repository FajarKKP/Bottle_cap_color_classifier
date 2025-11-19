# ---- Base image ----
FROM python:3.11-slim AS base

WORKDIR /app

# ---- System dependencies (minimal) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Poetry ----
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir poetry

# ---- Copy only dependency files first ----
COPY pyproject.toml poetry.lock ./

# ---- Install dependencies WITHOUT venv ----
# And use CPU-only PyTorch wheels
RUN poetry config virtualenvs.create false \
    && poetry config installer.no-binary ':all:' false \
    && poetry install --no-interaction --no-ansi --no-root

# ---- Copy app source ----
COPY . .

# ---- Default runtime entrypoint ----
ENTRYPOINT ["python", "-m", "bsort.cli"]
