FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    python3-dev \
    flac \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY vh_voice_analyzer_rest_agent.py .

# Expose the port the agent runs on
EXPOSE 8004

# Run the agent
CMD ["python", "vh_voice_analyzer_rest_agent.py"] 