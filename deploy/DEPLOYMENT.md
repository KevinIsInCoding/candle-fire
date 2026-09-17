# Stage 1 deployment record — candle-fire on EC2

**Deployed:** 2026-09-17 · **URL:** https://candle-fire.candlefireai.org · **Status:** live

Apps are served on **per-app subdomains** (chosen over path-prefixes: Gradio
streams SSE/websockets, which are fragile under a sub-path). `candlefireai.org`
and `www` redirect to the app subdomain; future `beacon` gets its own
`beacon.candlefireai.org`.

This is the record of the actual Stage 1 lift-and-shift (single CPU EC2 box, no
Docker, direct Anthropic API). For the rationale see
[`../docs/aws-migration-6pager.md`](../docs/aws-migration-6pager.md); for the
generic runbook see [`AWS-SETUP.md`](AWS-SETUP.md).

## What runs where

```
User ──443 HTTPS──► Caddy ──127.0.0.1:7860──► Gradio (app.py, systemd service)
                    (auto Let's Encrypt)       Chroma + KG + BioLORD + reranker
                                               data/ on persistent EBS
                                                      └──► Anthropic API (direct)
```

No Docker, no container image, no ECR. Code is pulled with `git clone`; Python
deps installed with `uv sync`; the process is run by `systemd`; TLS is handled by
Caddy. The runtime data (~1.4 GB) is **not** in git — the app downloads it from a
public HuggingFace dataset on first boot and it then persists on EBS.

## Infrastructure

| Item | Value |
|---|---|
| Region | us-east-1 |
| AWS account | 046451670096 |
| Instance | Ubuntu 26.04 LTS, `aarch64` (Graviton, `t4g`-class) |
| Elastic IP | 18.205.4.172 |
| Root disk | 30 GiB gp3 |
| App URL | `candle-fire.candlefireai.org` (root + `www` redirect here) |
| Domain / DNS | Route 53 zone `Z03758543IOFY1T6M2GH3`; A records: `candle-fire`, `@`, `www` → EIP |
| Security group | 443 + 80 from `0.0.0.0/0`; 22 from admin IP; 7860 closed |
| SSH | `ssh -i ~/.ssh/candle-fire.pem ubuntu@18.205.4.172` (alias `cf-ssh`) |
| Deploy branch | `main-aws` (GitHub default branch) |

## What was done

### Step 5 — DNS (Route 53, via awscli)
`UPSERT` A records in hosted zone `Z03758543IOFY1T6M2GH3`, all → `18.205.4.172` (TTL 300):
- `candle-fire.candlefireai.org` (the app)
- `candlefireai.org` and `www.candlefireai.org` (redirect to the app subdomain)
- `*.candlefireai.org` (wildcard — any future app subdomain, e.g. `beacon`, resolves with no new record)

```bash
aws route53 change-resource-record-sets --hosted-zone-id Z03758543IOFY1T6M2GH3 \
  --change-batch file://cf-dns.json
```

### Step 6 — provision the box (via SSH)
```bash
# on the box
git clone -b main-aws https://github.com/KevinIsInCoding/candle-fire.git /tmp/cf
sudo DOMAIN=candlefireai.org bash /tmp/cf/deploy/bootstrap.sh
```
`bootstrap.sh` installed: `uv` (0.12.15), `caddy` (2.11.4), a `candle` service
user, the app at `/opt/candle-fire` with its `.venv` (`uv sync`), the systemd
unit, and `/etc/caddy/Caddyfile` for `candlefireai.org`.

### Step 6 — secrets + start
`/opt/candle-fire/.env` (owner `candle`, `chmod 600`):
```
ANTHROPIC_API_KEY=<set>          # runtime LLM path (direct Anthropic API)
# HF_TOKEN not needed — the dataset repo is public
GRADIO_SERVER_NAME=127.0.0.1     # localhost only; Caddy is the public front door
GRADIO_SERVER_PORT=7860
CANDLE_LOG_LEVEL=INFO
```
```bash
sudo systemctl start candle-fire
```
First boot downloaded the corpus from the public HF dataset to EBS
(`_ensure_data()` in `app.py`) — one time; restarts skip it.

## Verification (all green)
- App: `curl -I http://127.0.0.1:7860` → 200; KG loaded 37,520 nodes / 22,575 edges.
- Public: `https://candle-fire.candlefireai.org` → 200; valid Let's Encrypt cert; `http` → HTTPS.
- Redirect: `candlefireai.org` and `www` → 302 → `https://candle-fire.candlefireai.org`.
- LLM: Anthropic key validated (HTTP 200 against `/v1/messages`).

## Operations

```bash
cf-ssh                                        # SSH in

sudo systemctl status candle-fire             # health
sudo journalctl -u candle-fire -f             # app logs
sudo journalctl -u caddy -f                   # TLS / proxy logs
sudo systemctl restart candle-fire            # restart (data persists — no re-download)

# Deploy new code (after merging to main-aws):
cd /opt/candle-fire
sudo -u candle git pull
sudo -u candle /usr/local/bin/uv sync
sudo systemctl restart candle-fire

# Backup: take a weekly EBS snapshot of the root volume (rebuild ≈ 15 min).
```

## Cost (actual shape, us-east-1)
`t4g.large` 24/7 (~$30–49/mo) + 30 GB gp3 (~$2.4) + Route 53 zone (~$0.5) +
Anthropic tokens (~$5–15). Elastic IP is free while attached. **~$40–55/mo**
on-demand; ~$25–35 with a 1-yr Savings Plan.

## Outstanding / follow-ups
- [ ] **Rotate the Anthropic key** — it was pasted in a chat transcript during
      deploy. Create a new key in the Anthropic console, update `.env`, restart.
      (HF token was never needed.)
- [ ] Point the HuggingFace deploy flow at `main-aws` (or retire it — Stage 1
      moves off HuggingFace).
- [ ] Optional: weekly EBS snapshot schedule; a billing budget/alert.
- [ ] Stage 2 (Bedrock/HIPAA) and Stage 3 (self-hosted GPU) remain deferred.
