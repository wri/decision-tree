#!/bin/bash

export UV_NATIVE_TLS=true
export SSL_CERT_DIR=/etc/ssl/certs
uv sync --native-tls
uv lock --upgrade # -U
source .venv/bin/activate
