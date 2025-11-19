# ---- Base image ----
FROM python:3.11-slim

WORKDIR /app

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Poetry ----
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir poetry

# ---- Copy dependency files ----
COPY pyproject.toml poetry.lock ./

# ---- Install CPU-only PyTorch first ----
RUN pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu

# ---- Disable Poetry virtualenv and install remaining dependencies ----
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi --no-root --without torch

# ---- Copy project files ----
COPY . .

# ---- Default entrypoint ----
ENTRYPOINT ["python", "-m", "bsort.cli"]
