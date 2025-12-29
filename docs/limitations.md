# Limitations and Scale Notes

This project currently targets single-instance deployment for internal lab use.

Known limitations:
- SQLite is local to the API host and not safe for multi-instance writes.
- Preflight caching is in-memory and does not share state across instances.
- The worker loop runs inside a single process and polls locally.
- Artifact paths assume a shared filesystem that is mounted on the API host.

Scaling path (future work):
- Migrate runs/queue storage to Postgres.
- Replace in-memory cache with Redis.
- Use a durable queue (e.g., Redis/RQ, Celery, or SQS) for run orchestration.
- Store artifacts in object storage (S3/MinIO) and serve via signed URLs.
