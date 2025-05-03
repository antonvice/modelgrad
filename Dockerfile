# Use an official Python runtime as a parent image
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME="0.0.0.0" \
    HF_HOME="/app/.cache/huggingface" \
    TRANSFORMERS_CACHE="/app/.cache/huggingface" \
    HF_HUB_CACHE="/app/.cache/huggingface/hub"

# Install system dependencies if needed
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire app directory (containing __init__.py, hf_app.py, task_config.py)
COPY ./app /app/app

EXPOSE 7860

# Define the command to run the application using the module syntax
# This correctly handles relative imports within the 'app' package
CMD ["python", "-m", "app.hf_app"]