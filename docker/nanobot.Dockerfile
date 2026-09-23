ARG TASK_TOOLS_IMAGE=indic-harness-task-tools:pilot-v13
FROM ${TASK_TOOLS_IMAGE}

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

RUN python -m pip install --no-cache-dir "nanobot-ai==${NANOBOT_VERSION}"

CMD ["sleep", "infinity"]
