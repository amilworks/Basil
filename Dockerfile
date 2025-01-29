# Use an official Python base image with a suitable version
FROM python:3.10-slim

# Disable Python’s .pyc files, enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set a working directory in the container
WORKDIR /app

# (Optional) Install system-level dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file to the container
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code
COPY . /app

# Expose FastAPI's default port
EXPOSE 8000

# Default command for running the FastAPI app
# NOTE: If you want to run Celery within the same image, you'll typically define separate entry points
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
