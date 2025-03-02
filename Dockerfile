FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required for DuckDB
RUN apt-get update && apt-get install -y \
    curl \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY lineup.py teams.py fapi.py games.py utils.py ./

# Set environment variable for the port
ENV PORT=8000

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "fapi:app", "--host", "0.0.0.0", "--port", "8000"]