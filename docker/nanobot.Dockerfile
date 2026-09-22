FROM python:3.12-slim

ARG NANOBOT_VERSION=0.3.5
LABEL org.opencontainers.image.title="Indic Harness Bench NanoBot runtime"
LABEL org.opencontainers.image.version="${NANOBOT_VERSION}"
LABEL org.opencontainers.image.source="https://github.com/HKUDS/nanobot"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOME=/state
ENV NANOBOT_HOME=/state
ENV NANOBOT_WORKSPACE=/workspace

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git sqlite3 zip unzip coreutils \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir "nanobot-ai==${NANOBOT_VERSION}"

CMD ["sleep", "infinity"]
