FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Set environment variable for model path
ENV SERVING_MODEL_DIR=wildan14ar-pipeline/serving_model/1748013612

# Expose port (Railway uses $PORT)
EXPOSE 8080

# Start Flask app via Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
