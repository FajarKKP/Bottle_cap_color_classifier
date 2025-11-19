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

# ---- Install CPU-only PyTorch first so that poertry skip it ----
RUN pip install --no-cache-dir torch==2.9.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

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
