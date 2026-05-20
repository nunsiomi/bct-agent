FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/task_b

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for layer caching.
COPY task_b/requirements.txt /app/task_b/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/task_b/requirements.txt

# Shared module + persona_signals.csv.
COPY shared /app/shared

# Retrieval catalog (consumed by retrieval_node.py).
COPY data_prep/artifacts /app/data_prep/artifacts

# Task B source (main.py + agent/).
COPY task_b /app/task_b

# /app on PYTHONPATH so `from shared.utils import …` and
# `from task_b.agent.X import …` both resolve.
ENV PYTHONPATH=/app

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
