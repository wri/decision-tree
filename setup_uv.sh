#!/bin/bash

uv lock --upgrade # -U
export UV_NATIVE_TLS=true
export SSL_CERT_DIR=/etc/ssl/certs
uv sync --native-tls
source .venv/bin/activate
