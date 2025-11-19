# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy poetry files first for dependency caching
COPY pyproject.toml poetry.lock* /app/

# Upgrade pip and install poetry
RUN pip install --upgrade pip && pip install poetry

# Install dependencies without creating a virtual environment
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy the rest of the project
COPY . /app

# Set entrypoint for CLI
ENTRYPOINT ["python", "-m", "bsort.cli"]
