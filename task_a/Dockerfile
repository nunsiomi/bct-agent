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
COPY task_a/requirements.txt /app/task_a/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/task_a/requirements.txt

# Source + data, in dependency order for cache hits.
COPY datasets       /app/datasets
COPY core           /app/core
COPY data_pipeline  /app/data_pipeline
COPY data_prep      /app/data_prep
COPY task_a         /app/task_a

# Build data artifacts that aren't tracked in git:
#   - catalog.json IS tracked, but we rebuild defensively so the image is self-contained.
#   - user_histories.csv (Phase 5 grounding) and vector_index.npz (Phase 4 retrieval)
#     are gitignored binaries -- must be regenerated inside the image.
RUN python -m data_pipeline.build_catalog && \
    python -m data_pipeline.build_user_histories && \
    python -m data_pipeline.build_vector_index

EXPOSE 8001
CMD ["uvicorn", "task_a.main:app", "--host", "0.0.0.0", "--port", "8001"]
