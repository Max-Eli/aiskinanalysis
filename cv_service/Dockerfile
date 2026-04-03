FROM python:3.11-slim

WORKDIR /app

# Only what we actually need — no OpenGL required for server-side CV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the SegFormer face-parsing model into the image at build time.
# This avoids a 300MB download on first request and prevents Railway timeouts.
RUN python -c "\
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation; \
print('Downloading face-parsing model...'); \
SegformerImageProcessor.from_pretrained('jonathandinu/face-parsing'); \
SegformerForSemanticSegmentation.from_pretrained('jonathandinu/face-parsing'); \
print('Model cached.')"

EXPOSE 8000
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
