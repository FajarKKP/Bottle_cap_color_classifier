# ---- Base image ----
FROM python:3.11-slim

# ---- Set working directory ----
WORKDIR /app

# ---- System dependencies (minimal) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Upgrade pip and install Poetry ----
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir poetry

# ---- Copy only dependency files first ----
COPY pyproject.toml poetry.lock ./

# ---- Install only runtime dependencies without venv ----
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi --no-root

# ---- Force CPU-only PyTorch (optional, safe) ----
RUN pip install --no-cache-dir torch==2.9.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# ---- Copy application code ----
COPY . .

# ---- Default runtime entrypoint ----
ENTRYPOINT ["python", "-m", "bsort.cli"]
