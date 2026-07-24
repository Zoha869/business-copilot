FROM python:3.11-slim

WORKDIR /app

# build-essential: some deps (fastembed/onnxruntime, PyMuPDF) pull wheels that
# still need a compiler on slim images for any sdist fallback.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Fly injects PORT; default kept for local docker run testing.
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
