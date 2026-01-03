# HPG setup for NicheRunner

This guide describes a production-oriented setup for running NicheRunner on HPG.
It avoids test files and uses real HPG paths for data, runs, and registries.

## 1) Choose shared directories

Pick a base directory that SLURM jobs can read and write, for example:

- `/blue/<group>/<user>/nicherunner`
- `/blue/<group>/<user>/data`
- `/blue/<group>/<user>/runs`

Suggested layout:

```
/blue/<group>/<user>/
  nicherunner/              # repo checkout (or a deploy location)
  data/                     # input datasets (h5ad, metadata)
  runs/                     # run outputs (artifacts, logs)
  registries/               # datasets.json, presets.json (optional shared copies)
  conda/envs/nicherunner/   # pipeline conda env
```

## 2) Create the pipeline conda environment

Use a shared conda env so SLURM jobs can activate it.

```bash
module load conda
conda create -p /blue/<group>/<user>/conda/envs/nicherunner python=3.10 -y
conda activate /blue/<group>/<user>/conda/envs/nicherunner
pip install --upgrade pip
```

Install core pipeline dependencies:

```bash
pip install cell2location scvi-tools scanpy anndata pandas numpy scipy scikit-learn seaborn matplotlib tqdm kneed scikit-posthocs statsmodels pyarrow papermill
```

Optional (for PDF report output):

```bash
module load pandoc
```

GPU nodes:
If you run on GPU partitions, install torch with CUDA:

```bash
conda install -y -c pytorch -c nvidia pytorch pytorch-cuda=11.8
```

## 3) Configure SLURM defaults in presets

Ensure presets specify the shared conda env and HPG resources. Example:

```json
"slurm": {
  "enabled": true,
  "conda_env": "/blue/<group>/<user>/conda/envs/nicherunner",
  "account": "<hpg-account>",
  "partition": "<partition>",
  "qos": "<qos>",
  "time": "96h",
  "mem": "400gb",
  "cpus_per_task": 8
}
```

If your HPG setup requires `module load conda` in jobs:

```bash
export SLURM_USE_MODULE_CONDA=1
```

## 4) Create the dataset registry (production data)

Use `registries/datasets.json` (or a shared file path via env var) to point to
real HPG data. Example entry:

```json
{
  "id": "ibd_cosmx_prod",
  "label": "IBD CosMx Production",
  "organ": "colon",
  "platform": "cosmx",
  "staged_path": "/blue/<group>/<user>/data/GSE234713_CosMx_combined.h5ad",
  "cell_metadata_path": "/blue/<group>/<user>/data/GSE234713_CosMx_cell_metadata.csv.gz",
  "reference_h5ad_path": "/blue/<group>/<user>/data/combined_10x_reference_final.h5ad",
  "recommended_preset": "ibd_cosmx_k4"
}
```

Then select the dataset in the UI or pass `dataset_id` in the run config.

## 5) Environment variables for the API (production)

Set these on the API host:

```bash
export RUNS_DIR=/blue/<group>/<user>/runs
export ARTIFACT_ROOTS=/blue/<group>/<user>/runs,/blue/<group>/<user>/data
export DATASETS_REGISTRY_PATH=/blue/<group>/<user>/registries/datasets.json
export PRESETS_DIR=/blue/<group>/<user>/nicherunner/presets
export DB_PATH=/blue/<group>/<user>/nicherunner/runs.db

export SESSION_SECRET="<strong-secret>"
export BASIC_AUTH_USER="<user>"
export BASIC_AUTH_PASS="<pass>"
```

If the API host does not have `anndata` or `pandas`, enable SLURM fallback for
join-key validation:

```bash
export PREFLIGHT_SLURM_FALLBACK=true
```

## 6) API deployment options

Option A: API runs on HPG login node (simplest, not ideal for long-running services).

```bash
cd apps/api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Option B: API runs on a VM, SLURM runs on HPG.
Use the SSH submitter described in `docs/SSH_SUBMITTER.md`.

## 7) Web UI environment

Point the UI at the API:

```bash
export NEXT_PUBLIC_API_BASE=http://<api-host>:8000
```

## 8) Production run behavior

The pipeline uses the scRNA h5ad for annotations and the spatial h5ad for
cell2location, NMF, and downstream analysis. The NMF stage now writes:

```
<output_dir>/cosmx_with_nmf.h5ad
```

This file includes `NMF_factor` in `.obs` and is used by the RCausalMGM stage
by default. RCausalMGM outputs go to:

```
<output_dir>/rcausal_mgm/
```

So production runs stay self-contained under the run output directory.

## 9) Common issues

- "Path outside ARTIFACT_ROOTS": add the data root to `ARTIFACT_ROOTS`.
- "Post-NMF notebook not found": ensure `pipeline_assets/IBD_Post_NMF_Analysis.ipynb`
  exists on the run host.
- "RCausalMGM script not found": ensure `pipeline_assets/IBD_RCausalMGM_Preparation.py`
  exists on the run host.
- "Join-key validation missing deps": install `anndata` and `pandas` on the API host
  or enable SLURM fallback.

## 10) Quick checklist

1. Conda env created on HPG with pipeline dependencies.
2. Data paths stored in the dataset registry.
3. `RUNS_DIR` and `ARTIFACT_ROOTS` set to shared HPG locations.
4. Preset `slurm.conda_env` points to the shared conda env.
5. API reachable from the web UI.
