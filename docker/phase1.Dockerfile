FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends git sqlite3 zip unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir pytest

CMD ["sleep", "infinity"]
