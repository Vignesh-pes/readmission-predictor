# --------------------------------------------------
# Base Image
# --------------------------------------------------
FROM python:3.10-slim

# --------------------------------------------------
# Environment Variables
# --------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# --------------------------------------------------
# Set Working Directory
# --------------------------------------------------
WORKDIR /app

# --------------------------------------------------
# Install System Dependencies (if required)
# --------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# Copy Dependency Files
# --------------------------------------------------
COPY requirements.txt .

# --------------------------------------------------
# Install Python Dependencies
# --------------------------------------------------
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------
# Copy Application Source Code
# --------------------------------------------------
COPY src/ ./src/
COPY artifacts/ ./artifacts/

# --------------------------------------------------
# Expose Application Port
# --------------------------------------------------
EXPOSE 8000

# --------------------------------------------------
# Run Application
# --------------------------------------------------
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
