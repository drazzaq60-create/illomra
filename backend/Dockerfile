# Illomra backend — FastAPI + Gemini + Chroma. Embeddings run on Google's API,
# so no torch/sentence-transformers here — the image stays small for free hosting.
#   docker build -t illomra-backend .
#   docker run -p 10000:10000 -e GOOGLE_API_KEY=... illomra-backend
FROM python:3.11-slim

WORKDIR /app

# Install deps first so Docker caches this layer between code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Where the Chroma store + conversation files live. On a host WITH a persistent
# disk mounted here (e.g. a Railway/Fly volume at /data) documents survive
# restarts; on a free host without one, /data is ephemeral and resets on restart.
RUN mkdir -p /data
ENV PERSIST_DIR=/data/chroma_db
# The host provides $PORT (Koyeb/Render inject it); this is the local default.
ENV PORT=10000
EXPOSE 10000

# ONE worker only: Chroma's SQLite is single-writer and the engine is a single
# shared in-process object. Scale vertically, not horizontally.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers 1"]
