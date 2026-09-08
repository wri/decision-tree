FROM mcr.microsoft.com/devcontainers/python:3.12

RUN sudo apt-get update \
    && sudo apt-get install -y --no-install-recommends \
        gdal-bin \
        libgdal-dev \
        proj-bin \
        libproj-dev \
    && sudo rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

ENV UV_NATIVE_TLS=true
ENV GDAL_CONFIG=/usr/bin/gdal-config
