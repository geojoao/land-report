# =====================================================================
# Stage 1: Builder - Install dependencies
# =====================================================================
FROM python:3.10-slim-bookworm

WORKDIR /app

# Install system dependencies required for building geospatial libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    gcc \
    git \
    libgdal-grass \
    && rm -rf /var/lib/apt/lists/*

# Install a recent version of uv and sync Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir "uv>=0.1.17"
RUN uv sync


# Copy your application's code and data into the container
COPY bacen_report.py .
COPY locks.xml .

# Command to run the script when the container launches
# Use the python interpreter from the virtual environment created by `uv sync`.
CMD [".venv/bin/python", "bacen_report.py"]