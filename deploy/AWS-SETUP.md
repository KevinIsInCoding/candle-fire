# Stage 1 — AWS provisioning runbook (self-serve)

End-to-end steps to stand up candle-fire on one EC2 box. Console-first (best for
a fresh account); a CLI variant is at the bottom. Region assumed **us-east-1**.

Fill these in before you start:

| Placeholder | Your value |
|---|---|
| `DOMAIN` | `candlefireai.org` (serve on the root; optionally also `www`) |
| `MY_IP` | your public IP — run `curl ifconfig.me` |
| `ANTHROPIC_API_KEY` | your Claude API key — **entered on the box only, never in this file or in chat** |
| `HF_TOKEN` | read token (the data repo is private) — **box only, never committed** |

> Secrets live only in `/opt/candle-fire/.env` on the instance (gitignored,
> `chmod 600`). Do not paste them into docs, commits, or chat.

---

## 1. SSH key pair
EC2 → **Key Pairs** → *Create key pair* → name `candle-fire`, type RSA, format
`.pem` → download. Then locally:
```bash
mv ~/Downloads/candle-fire.pem ~/.ssh/ && chmod 400 ~/.ssh/candle-fire.pem
```

## 2. Security group
EC2 → **Security Groups** → *Create* → name `candle-fire-sg`. Inbound rules:

| Type | Port | Source |
|---|---|---|
| HTTPS | 443 | `0.0.0.0/0` |
| HTTP | 80 | `0.0.0.0/0` |
| SSH | 22 | `MY_IP/32` |

Leave outbound at default (all allowed).

## 3. Launch the instance
EC2 → **Instances** → *Launch instances*:
- **Name:** `candle-fire`
- **AMI:** Ubuntu Server 24.04 LTS — **pick the `arm64` image** for `t4g.large`
  (or the `x86_64` image if you choose `t3.large`)
- **Instance type:** `t4g.large` (recommended) or `t3.large`
- **Key pair:** `candle-fire`
- **Network / firewall:** select existing SG → `candle-fire-sg`
- **Storage:** change root volume to **30 GiB, gp3**
- *Launch instance*

## 4. Elastic IP (stable address)
EC2 → **Elastic IPs** → *Allocate* → then *Actions → Associate* → to the
`candle-fire` instance. Note the IP — call it `ELASTIC_IP`.

## 5. DNS
Apps are served on **per-app subdomains** (see `Caddyfile`). Add **A records**
pointing at `ELASTIC_IP`:

| Host | Type | Value | Purpose |
|---|---|---|---|
| `candle-fire` | A | `ELASTIC_IP` | the app: `candle-fire.candlefireai.org` |
| `@` (root) | A | `ELASTIC_IP` | redirects to the app subdomain |
| `www` | A | `ELASTIC_IP` | redirects to the app subdomain |

Tip: a single wildcard `*` A record → `ELASTIC_IP` covers every future app
subdomain (e.g. `beacon`) without adding records each time.

Give it a minute: `dig +short candle-fire.candlefireai.org` should return
`ELASTIC_IP`. DNS must resolve *before* the first HTTPS hit, or Caddy can't
issue the cert.

## 6. Deploy on the box
```bash
ssh -i ~/.ssh/candle-fire.pem ubuntu@ELASTIC_IP

# on the box:
git clone -b main-aws https://github.com/KevinIsInCoding/candle-fire.git /tmp/cf
sudo DOMAIN=candlefireai.org bash /tmp/cf/deploy/bootstrap.sh

sudo nano /opt/candle-fire/.env        # set ANTHROPIC_API_KEY (+ HF_TOKEN if private)
sudo systemctl start candle-fire
sudo journalctl -u candle-fire -f      # first boot pulls ~1.4 GB to EBS (one time)
```
Wait for the log line showing Gradio running on `127.0.0.1:7860`.

## 7. Verify
Browse `https://DOMAIN` (Caddy issues the TLS cert on the first HTTPS hit) and
run a real question end-to-end. Keep the HuggingFace Space alive as a fallback
for a few days before decommissioning.

---

## Troubleshooting
- **Cert not issued:** DNS must resolve to `ELASTIC_IP` *before* the first hit,
  and port 80 must be open. Check `sudo journalctl -u caddy -f`.
- **Service won't start / OOM:** `journalctl -u candle-fire -e`. If RAM is the
  issue, stop the instance and change type to `t4g.xlarge` (16 GB).
- **HF download fails:** repo is private → add `HF_TOKEN` to `.env`, then
  `sudo systemctl restart candle-fire`.

---

## Appendix: CLI variant (if you prefer)
Requires the AWS CLI configured (`aws configure`). Replace `ami-xxxx` with the
current Ubuntu 24.04 arm64 AMI for your region (SSM lookup below).
```bash
AMI=$(aws ssm get-parameters --region us-east-1 \
  --names /aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id \
  --query 'Parameters[0].Value' --output text)

SG=$(aws ec2 create-security-group --group-name candle-fire-sg \
  --description "candle-fire" --query GroupId --output text)
aws ec2 authorize-security-group-ingress --group-id $SG --protocol tcp --port 443 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id $SG --protocol tcp --port 80  --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id $SG --protocol tcp --port 22  --cidr $(curl -s ifconfig.me)/32

aws ec2 run-instances --image-id $AMI --instance-type t4g.large \
  --key-name candle-fire --security-group-ids $SG \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=candle-fire}]'
```
Then allocate/associate an Elastic IP, add the DNS record, and follow §6.
