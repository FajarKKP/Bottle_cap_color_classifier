# ---- Stage 1: Builder (install only non-heavy runtime deps) ----
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install pip + Poetry
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "poetry>=1.5.1"

# Copy only dependency descriptors
COPY pyproject.toml poetry.lock ./

# Configure Poetry: no virtualenv, only install main group
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main \
    && rm -rf /root/.cache/pypoetry/* /root/.cache/pip/*


# ---- Stage 2: Final Runtime Image ----
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python deps
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project files
COPY . .

# Install CPU-only torch 
RUN pip install --no-cache-dir torch==2.9.1 \
    --index-url https://download.pytorch.org/whl/cpu

ENTRYPOINT ["python", "-m", "bsort.cli"]
