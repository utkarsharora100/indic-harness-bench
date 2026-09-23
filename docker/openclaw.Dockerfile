FROM node:24.16.0-bookworm-slim AS node-runtime

ARG TASK_TOOLS_IMAGE=indic-harness-task-tools:pilot-v13
FROM ${TASK_TOOLS_IMAGE}

COPY --from=node-runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=node-runtime /usr/local/lib/node_modules /usr/local/lib/node_modules

ARG OPENCLAW_VERSION=2026.9.5
LABEL org.opencontainers.image.title="Indic Harness Bench OpenClaw runtime"
LABEL org.opencontainers.image.version="${OPENCLAW_VERSION}"
LABEL org.opencontainers.image.source="https://github.com/openclaw/openclaw"

ENV HOME=/state
ENV OPENCLAW_STATE_DIR=/state
ENV OPENCLAW_CONFIG_PATH=/state/openclaw.json
ENV OPENCLAW_AGENT_WORKSPACE_DIR=/workspace

WORKDIR /workspace

RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && npm install --global "openclaw@${OPENCLAW_VERSION}" \
    && npm cache clean --force

CMD ["sleep", "infinity"]
