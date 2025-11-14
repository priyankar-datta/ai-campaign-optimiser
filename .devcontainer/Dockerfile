ARG VARIANT="3.10"
FROM mcr.microsoft.com/devcontainers/python:0-${VARIANT}

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git wget curl ca-certificates libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace
RUN python3 -m pip install --upgrade pip setuptools wheel
