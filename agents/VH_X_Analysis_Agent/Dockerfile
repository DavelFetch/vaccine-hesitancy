FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY x_analysis_rest_agent.py .
COPY x_analysis_models.py .

# Expose the port the agent runs on
EXPOSE 8001

# Run the agent
CMD ["python", "x_analysis_rest_agent.py"]