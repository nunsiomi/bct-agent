"""Offline data pipeline: ingest -> clean -> build artifacts the services consume.

Run-once, deterministic, idempotent. The Docker services never run this; they
read its outputs from `data_prep/artifacts/`.
"""
