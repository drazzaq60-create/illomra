# Illomra backend — FastAPI + local embeddings + Chroma.
# Built for Hugging Face Spaces (Docker Space, port 7860) but works anywhere:
#   docker build -t studymind-backend .
#   docker run -p 7860:7860 -e GOOGLE_API_KEY=... -v studymind_data:/data studymind-backend
FROM python:3.11-slim

WORKDIR /app

# Install deps first so Docker caches this layer between code changes.
COPY requirements.lock.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt

# Pre-download the embedding model at BUILD time so cold starts don't depend
# on huggingface.co being reachable (and don't add 30-60s to first boot).
ENV HF_HOME=/data/hf_cache
RUN HF_HOME=/tmp/hf_build python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')" \
    && mkdir -p /data/hf_cache && cp -r /tmp/hf_build/. /data/hf_cache/ || true

COPY . .

# /data is the persistent volume on HF Spaces / Fly.io — the vector store and
# model cache must live there or every deploy wipes the teacher's documents.
ENV PERSIST_DIR=/data/chroma_db
ENV PORT=10000

EXPOSE 10000

# ONE worker only: Chroma's SQLite is single-writer and the engine is a
# single shared in-process object. Scale vertically, not horizontally.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers 1"]
