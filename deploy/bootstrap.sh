#!/usr/bin/env bash
#
# Stage 1 bootstrap for candle-fire on a fresh Ubuntu 24.04 EC2 box.
# Lift-and-shift only: single CPU box, Anthropic API unchanged, Caddy TLS.
# (Bedrock/HIPAA = Stage 2, self-hosted GPU = Stage 3 — NOT installed here.)
#
# Run as root on the box:
#   sudo DOMAIN=candle-fire.example.com bash deploy/bootstrap.sh
#
# Re-runnable: skips clone/user/.env if they already exist.

set -euo pipefail

# ── Config (override via env) ────────────────────────────────────────────────
DOMAIN="${DOMAIN:?set DOMAIN, e.g. DOMAIN=candle-fire.example.com}"
REPO_URL="${REPO_URL:-https://github.com/KevinIsInCoding/candle-fire.git}"
BRANCH="${BRANCH:-main-aws}"   # deploy branch (holds deploy/ + AWS runbook)
APP_DIR="${APP_DIR:-/opt/candle-fire}"
SVC_USER="${SVC_USER:-candle}"

echo ">> candle-fire Stage 1 bootstrap  (domain=$DOMAIN  app_dir=$APP_DIR)"

# ── System packages ──────────────────────────────────────────────────────────
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y git curl gpg debian-keyring debian-archive-keyring apt-transport-https

# ── Caddy (official apt repo: installs caddy binary + systemd unit + /etc/caddy) ─
if ! command -v caddy >/dev/null 2>&1; then
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    | tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  apt-get update
  apt-get install -y caddy
fi

# ── uv (system-wide, so the systemd service can find it) ─────────────────────
if [ ! -x /usr/local/bin/uv ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
fi

# ── Service user + code ──────────────────────────────────────────────────────
id -u "$SVC_USER" &>/dev/null || useradd --system --create-home --shell /usr/sbin/nologin "$SVC_USER"
if [ ! -d "$APP_DIR/.git" ]; then
  git clone -b "$BRANCH" "$REPO_URL" "$APP_DIR"
fi
chown -R "$SVC_USER:$SVC_USER" "$APP_DIR"

# ── Python deps (as the service user) ────────────────────────────────────────
sudo -u "$SVC_USER" env PATH="/usr/local/bin:$PATH" bash -c "cd '$APP_DIR' && uv sync"

# ── .env (created once; you must fill in the API key) ────────────────────────
if [ ! -f "$APP_DIR/.env" ]; then
  cat > "$APP_DIR/.env" <<'EOF'
# REQUIRED — set your real key before starting the service.
ANTHROPIC_API_KEY=REPLACE_ME

# Only needed if the HF data repo (KevinIsCoding/candle-fire-data) is PRIVATE.
# HF_TOKEN=REPLACE_ME

# Bind Gradio to localhost only; Caddy is the public front door.
GRADIO_SERVER_NAME=127.0.0.1
GRADIO_SERVER_PORT=7860

CANDLE_LOG_LEVEL=WARNING
EOF
  chown "$SVC_USER:$SVC_USER" "$APP_DIR/.env"
  chmod 600 "$APP_DIR/.env"
  echo ">> Created $APP_DIR/.env — EDIT IT and set ANTHROPIC_API_KEY."
fi

# ── systemd unit ─────────────────────────────────────────────────────────────
cp "$APP_DIR/deploy/candle-fire.service" /etc/systemd/system/candle-fire.service
systemctl daemon-reload
systemctl enable candle-fire

# ── Caddy config ─────────────────────────────────────────────────────────────
sed "s/{{DOMAIN}}/$DOMAIN/g" "$APP_DIR/deploy/Caddyfile" > /etc/caddy/Caddyfile
systemctl reload caddy || systemctl restart caddy

cat <<EOF

────────────────────────────────────────────────────────────────────────────
Bootstrap complete. Remaining manual steps:

  1. Set the API key:      sudo nano $APP_DIR/.env    # ANTHROPIC_API_KEY=...
  2. Start the app:        sudo systemctl start candle-fire
     First boot downloads ~1.4 GB to EBS (one time). Watch it:
                           sudo journalctl -u candle-fire -f
  3. Point DNS: an A record for $DOMAIN -> this box's Elastic IP.
     Once DNS resolves, Caddy issues the TLS cert automatically on first hit.
  4. Verify:               https://$DOMAIN
────────────────────────────────────────────────────────────────────────────
EOF
