# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy package files
COPY . .

# Install git (needed for abydos installation from GitHub)
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install the package
RUN pip install -e .

# Set Python path
ENV PYTHONPATH=/app

# Default command to run Python REPL
CMD ["python"] 