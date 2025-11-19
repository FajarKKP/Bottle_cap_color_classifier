# ---- Base image ----
FROM python:3.11-slim

WORKDIR /app

# ---- Minimal system dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Upgrade pip ----
RUN pip install --no-cache-dir --upgrade pip

# ---- Install CPU-only PyTorch first so that poetry skip it ----
RUN pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu

# ---- Install Poetry ----
RUN pip install --no-cache-dir poetry

# ---- Copy dependency files ----
COPY pyproject.toml poetry.lock ./

# ---- Configure Poetry to avoid virtualenv ----
RUN poetry config virtualenvs.create false

# ---- Install remaining runtime dependencies ONLY ----
RUN poetry install --only main --no-interaction --no-ansi --no-root

# ---- Copy project files ----
COPY . .

# ---- Default entrypoint ----
ENTRYPOINT ["python", "-m", "bsort.cli"]
