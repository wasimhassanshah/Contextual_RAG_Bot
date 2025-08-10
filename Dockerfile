# Multi-stage build for Abu Dhabi Procurement RAG System
FROM python:3.10-slim-buster

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PHOENIX_PORT=6006
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    awscli \
    ffmpeg \
    libsm6 \
    libxext6 \
    unzip \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . /app

# Create necessary directories
RUN mkdir -p /app/Context_RAG/vectorstore && \
    mkdir -p /app/webui_interface && \
    mkdir -p /app/phoenix_interface && \
    mkdir -p /app/data

# Set proper permissions
RUN chmod +x /app/webui_interface/custom_webui.py && \
    chmod +x /app/phoenix_interface/phoenix_app.py

# Expose ports
EXPOSE 8501 6006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command - can be overridden
CMD ["streamlit", "run", "webui_interface/custom_webui.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Alternative commands (uncomment as needed):
# For Phoenix standalone:
# CMD ["python3", "phoenix_interface/phoenix_app.py"]

# For custom app.py:
# CMD ["python3", "app.py"]

# For development with both interfaces:
# CMD ["bash", "-c", "python3 phoenix_interface/phoenix_app.py & streamlit run webui_interface/custom_webui.py --server.port=8501 --server.address=0.0.0.0"]