# Deployment notes

## Docker (local)

```bash
cd deploy
cp .env.example .env
# edit .env to set BASIC_AUTH_* or AUTH_PASSWORD_HASH and SESSION_SECRET
docker compose up --build
```

The compose file runs a separate worker container and stores the SQLite DB under `runs/`.
`NEXT_PUBLIC_API_BASE` is baked into the web build, so rebuild the web image after changing it.

## Nginx reverse proxy

Use `deploy/nginx.conf` as a starting point and terminate TLS at the proxy.

## Production checklist

- Set `SESSION_SECRET` and `BASIC_AUTH_*` or `AUTH_PASSWORD_HASH`.
- Set `COOKIE_SECURE=true` when behind HTTPS.
- Set `ARTIFACT_ROOTS` to allowed filesystem roots (e.g., `/blue` on HPG).
- Keep `WORKER_ENABLED=false` in the API when running a separate worker process.
- Set `RUN_RETENTION_DAYS` and disk warning thresholds for cleanup/alerts.
- Mount `registries/` and set `DATASETS_REGISTRY_PATH` if using dataset registry endpoints.
- Set `PREFLIGHT_SLURM_FALLBACK=true` if you want preflight to submit a SLURM check when API dependencies are missing.
- Set `PREFLIGHT_CACHE_TTL_SECONDS` to control preflight cache duration.

## UF HiPerGator checklist (VM + login-node worker)

- Worker runs on a HiPerGator login node using the same UF account as the pipeline.
- VM <-> login node uses SSH (non-interactive): install a dedicated keypair, add the public key
  to `~/.ssh/authorized_keys`, and manage `known_hosts` to avoid MITM prompts.
- Ensure `/blue/...` is mounted on the VM and the worker, with matching UID/GID ownership so
  both can read/write run outputs and the shared `DB_PATH`.
- VM serves the web/API behind TLS; set `ALLOWED_ORIGINS`, `COOKIE_SECURE=true`, and ensure
  CSRF headers are sent by the UI.
- For SLURM jobs, set `slurm.conda_env`; enable `slurm.use_module_conda` (or `SLURM_USE_MODULE_CONDA=true`)
  only when the login node needs `module load conda`. Otherwise set `CONDA_BASE` so the job can
  source `conda.sh` directly.
