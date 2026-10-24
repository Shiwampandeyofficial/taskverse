FROM python:3.11-slim

# System build dependencies (chromadb aur numpy/scipy ke liye zaroori)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# Source code copy
COPY . .

EXPOSE 8080
ENV PORT=8080

# Agar entry file app.py hai toh app:app chalega
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "120", "app:app"]
