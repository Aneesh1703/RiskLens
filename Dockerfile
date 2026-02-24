# Use the official Python 3.11 slim image
# The slim image is substantially smaller than the standard image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# (libgomp1 is sometimes required by onnxruntime or scikit-learn on slim images)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that the application listens on
EXPOSE 8080

# Command to run the application using Uvicorn
# We use the $PORT environment variable which is dynamically provided by Cloud Run
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
