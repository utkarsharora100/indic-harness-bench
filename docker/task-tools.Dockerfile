FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git sqlite3 zip unzip coreutils \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir "pytest==8.4.2" "pandas==2.3.3"

CMD ["sleep", "infinity"]
