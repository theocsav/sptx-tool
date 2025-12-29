# NicheRunner Web

## Setup

```bash
npm install
```

## Run

```bash
npm run dev
```

Set `NEXT_PUBLIC_API_BASE` to point at the FastAPI backend. Login sets a secure cookie; the UI does not store credentials.

## Data input (MVP)

Provide HPG paths to a prepared bundle (h5ad + metadata). NIH/GEO ingestion is deferred.
