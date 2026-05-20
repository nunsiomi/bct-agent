FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/task_a

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for layer caching.
COPY task_a/requirements.txt /app/task_a/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/task_a/requirements.txt

# Shared module + persona_signals.csv.
COPY shared /app/shared

# Task A source (main.py + agent/).
COPY task_a /app/task_a

# /app on PYTHONPATH so `from shared.utils import …` and
# `from task_a.agent.X import …` both resolve.
ENV PYTHONPATH=/app

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
