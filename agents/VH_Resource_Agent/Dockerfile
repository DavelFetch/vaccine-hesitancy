# Use official Python image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the agent code
COPY . .

# Expose the default port (update if you use a different one)
EXPOSE 8002

# Set environment variables (optional, for production best practices)
ENV PYTHONUNBUFFERED=1

# Default command to run the agent (update as needed)
CMD ["python", "vaccine_resource_agent.py"]