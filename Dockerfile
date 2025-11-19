# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install poetry
RUN poetry install

# If not using poetry, you could also:
# RUN pip install ultralytics opencv-python numpy pyyaml

# Set entrypoint for CLI
ENTRYPOINT ["python", "-m", "bsort.cli"]
