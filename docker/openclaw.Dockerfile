FROM node:24.16-bookworm-slim

ARG OPENCLAW_VERSION=2026.9.5
LABEL org.opencontainers.image.title="Indic Harness Bench OpenClaw runtime"
LABEL org.opencontainers.image.version="${OPENCLAW_VERSION}"
LABEL org.opencontainers.image.source="https://github.com/openclaw/openclaw"

ENV HOME=/state
ENV OPENCLAW_STATE_DIR=/state
ENV OPENCLAW_CONFIG_PATH=/state/openclaw.json
ENV OPENCLAW_AGENT_WORKSPACE_DIR=/workspace

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git sqlite3 zip unzip coreutils python3 \
    && rm -rf /var/lib/apt/lists/* \
    && npm install --global "openclaw@${OPENCLAW_VERSION}" \
    && npm cache clean --force

CMD ["sleep", "infinity"]
