#!/usr/bin/env bash
set -euo pipefail

# Cross-platform TLS hints (only when helpful)
case "$(uname -s)" in
  Linux)
    [ -d "/etc/ssl/certs" ] && export SSL_CERT_DIR="/etc/ssl/certs" || true
    ;;
  Darwin)
    # Prefer Homebrew's OpenSSL bundle if present; otherwise rely on native TLS
    if   [ -f "/opt/homebrew/etc/openssl@3/cert.pem" ]; then export SSL_CERT_FILE="/opt/homebrew/etc/openssl@3/cert.pem"
    elif [ -f "/usr/local/etc/openssl@3/cert.pem"   ]; then export SSL_CERT_FILE="/usr/local/etc/openssl@3/cert.pem"
    fi
    ;;
esac

export UV_NATIVE_TLS=true

uv lock --upgrade
uv sync --native-tls

# Activate the environment created by uv
. .venv/bin/activate