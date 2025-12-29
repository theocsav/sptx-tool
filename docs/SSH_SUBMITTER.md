# SSH Submitter (HiPerGator)

This project assumes a split deployment on HiPerGator:
- The API and web UI run on a lab-managed VM.
- A lightweight submitter/worker runs on a login node and executes SLURM commands.

The VM talks to the login-node submitter over SSH. The API does not ship an SSH
submitter yet; use the configuration below for your submitter wrapper/service.

## Required configuration

Define these in your submitter service or wrapper:

- `SSH_HOST` Login node hostname.
- `SSH_PORT` SSH port (default 22).
- `SSH_USER` UF account that owns `/blue/...` data and submits jobs.
- `SSH_KEY_PATH` Path to the private key used by the VM.
- `SSH_KNOWN_HOSTS` Optional path to `known_hosts` (default `~/.ssh/known_hosts`).
- `SSH_REMOTE_WORKDIR` Working directory on the login node (e.g. `/blue/.../nicherunner`).
- `SSH_ALLOWED_ROOTS` Comma-separated allowlist for paths (match API `ARTIFACT_ROOTS`).

Example (VM-side `.env`):

```bash
SSH_HOST=hpg-login.example.ufl.edu
SSH_PORT=22
SSH_USER=ufuserid
SSH_KEY_PATH=/home/nicherunner/.ssh/nicherunner_hpg
SSH_KNOWN_HOSTS=/home/nicherunner/.ssh/known_hosts
SSH_REMOTE_WORKDIR=/blue/lab/ufuserid/nicherunner
SSH_ALLOWED_ROOTS=/blue/lab/ufuserid,/orange/lab/ufuserid
```

## SSH key setup (no host key prompts)

```bash
SSH_HOST=hpg-login.example.ufl.edu
SSH_USER=ufuserid
SSH_KEY=~/.ssh/nicherunner_hpg

install -m 700 -d ~/.ssh
ssh-keygen -t ed25519 -C "nicherunner-vm" -f "$SSH_KEY" -N ""
ssh-keyscan -H "$SSH_HOST" >> ~/.ssh/known_hosts

cat "${SSH_KEY}.pub" | ssh -o StrictHostKeyChecking=yes "$SSH_USER@$SSH_HOST" \
  "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

If password authentication is disabled, install the public key via your UF-approved method.

## Minimal smoke test

Assumes:
- Dataset registry is updated.
- API is running on the VM.
- Submitter is running on the login node and can submit SLURM jobs.

1) Register a dataset manifest (run in the pipeline conda env):

```bash
python scripts/generate_dataset_manifest.py \
  --id smoke_cosmx \
  --label "Smoke CosMx" \
  --organ colon \
  --platform cosmx \
  --h5ad /blue/.../sample.h5ad \
  --metadata /blue/.../metadata.csv.gz \
  --registry registries/datasets.json
```

2) Login, preflight, submit a run:

```bash
API_BASE=http://vm-host:8000
USER=youruser
PASS=yourpass
COOKIE=$(mktemp)

CSRF=$(curl -s -c "$COOKIE" -H "Content-Type: application/json" \
  -X POST "$API_BASE/auth/login" \
  -d "{\"username\":\"$USER\",\"password\":\"$PASS\"}" \
  | python - <<'PY'
import json,sys
print(json.load(sys.stdin).get("csrf_token",""))
PY
)

cat > /tmp/nicherunner_smoke.json <<'JSON'
{
  "run_name": "smoke-run",
  "preset_path": "presets/ibd_cosmx_k4.json",
  "config": {
    "dataset_id": "smoke_cosmx",
    "stages": ["cell2loc_nmf"],
    "slurm": {
      "enabled": true,
      "conda_env": "nicherunner",
      "account": "your_account",
      "partition": "your_partition",
      "qos": "your_qos"
    }
  },
  "submit": true
}
JSON

curl -s -b "$COOKIE" -H "X-CSRF-Token: $CSRF" -H "Content-Type: application/json" \
  -X POST "$API_BASE/runs/preflight" \
  -d @/tmp/nicherunner_smoke.json | python -m json.tool

RUN_ID=$(curl -s -b "$COOKIE" -H "X-CSRF-Token: $CSRF" -H "Content-Type: application/json" \
  -X POST "$API_BASE/runs" \
  -d @/tmp/nicherunner_smoke.json \
  | python - <<'PY'
import json,sys
print(json.load(sys.stdin).get("id",""))
PY
)
```

3) Check status, logs, and artifacts:

```bash
curl -s -b "$COOKIE" "$API_BASE/runs/$RUN_ID" | python -m json.tool
curl -s -b "$COOKIE" "$API_BASE/runs/$RUN_ID/logs" | python -m json.tool
curl -s -b "$COOKIE" "$API_BASE/runs/$RUN_ID/artifacts" | python -m json.tool
```

Once the job finishes, confirm the report artifacts exist:

```bash
curl -s -b "$COOKIE" "$API_BASE/runs/$RUN_ID/artifacts?path=report" | python -m json.tool
```
