FROM python:3.12-slim

LABEL org.opencontainers.image.title="Indic Harness Bench model proxy"
LABEL org.opencontainers.image.version="phase1-pinned"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app
COPY runner/proxy.py /app/runner/proxy.py
COPY scripts/proxy_sidecar.py /app/scripts/proxy_sidecar.py
COPY runner/__init__.py /app/runner/__init__.py
COPY scripts/__init__.py /app/scripts/__init__.py

CMD ["python", "/app/scripts/proxy_sidecar.py"]
