# Stage 1 deployment — candle-fire on EC2

Lift-and-shift: one CPU EC2 box runs Gradio + Chroma + KG + embeddings, Caddy
terminates HTTPS, data lives on persistent EBS. LLM path is the direct Anthropic
API (unchanged). Bedrock/HIPAA (Stage 2) and self-hosted GPU (Stage 3) are not
part of this stage. Full rationale: [`../docs/aws-migration-6pager.md`](../docs/aws-migration-6pager.md).

## Files

| File | Purpose |
|---|---|
| `bootstrap.sh` | One-shot provisioner for a fresh Ubuntu 24.04 box (packages, uv, Caddy, service user, systemd, Caddy config). |
| `candle-fire.service` | systemd unit — runs `uv run python app.py`, restarts on crash, starts on boot. |
| `Caddyfile` | Reverse proxy `:443 -> 127.0.0.1:7860` with auto Let's Encrypt TLS. |

## Quick start

On a fresh **Ubuntu 24.04** `t4g.large` (2 vCPU / 8 GB) with a 30 GB gp3 root
volume and an Elastic IP:

```bash
git clone -b main-aws https://github.com/KevinIsInCoding/candle-fire.git /tmp/cf
sudo DOMAIN=candlefireai.org bash /tmp/cf/deploy/bootstrap.sh
sudo nano /opt/candle-fire/.env      # set ANTHROPIC_API_KEY
sudo systemctl start candle-fire
sudo journalctl -u candle-fire -f    # first boot pulls ~1.4 GB to EBS (once)
```

Point an A record for your domain at the Elastic IP; Caddy provisions the cert
on first HTTPS hit. Verify at `https://<your-domain>`.

## Security group

| Port | Source | Why |
|---|---|---|
| 443 | 0.0.0.0/0 | HTTPS |
| 80 | 0.0.0.0/0 | ACM HTTP-01 challenge + 80→443 redirect |
| 22 | your IP/32 | SSH |
| 7860 | — (closed) | Gradio is localhost-only behind Caddy |

## Operations

- Logs: `journalctl -u candle-fire -f`
- Restart: `sudo systemctl restart candle-fire` (data persists on EBS — no re-download)
- Update code: `cd /opt/candle-fire && sudo -u candle git pull && sudo -u candle /usr/local/bin/uv sync && sudo systemctl restart candle-fire`
- Backup: weekly EBS snapshot of the root volume is enough to rebuild in ~15 min.
