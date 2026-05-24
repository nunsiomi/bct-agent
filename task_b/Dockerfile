FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for layer caching.
COPY task_b/requirements.txt /app/task_b/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/task_b/requirements.txt

# Source + data, in dependency order for cache hits.
COPY datasets       /app/datasets
COPY core           /app/core
COPY data_pipeline  /app/data_pipeline
COPY data_prep      /app/data_prep
COPY task_b         /app/task_b

# Build the catalog + vector index inside the image so Task B retrieval works
# from a clean container start. (catalog.json is committed; vector_index.npz isn't.)
RUN python -m data_pipeline.build_catalog && \
    python -m data_pipeline.build_vector_index

EXPOSE 8002
CMD ["uvicorn", "task_b.main:app", "--host", "0.0.0.0", "--port", "8002"]
