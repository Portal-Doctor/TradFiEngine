# Use a slim Python image for efficiency
FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc and buffering stdout
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies for TA-Lib and other math libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Create the data directory for your SQLite database and logs
RUN mkdir -p data checkpoints

# Expose the Streamlit port
EXPOSE 8501

# We'll use Docker Compose to define which script runs in which container
