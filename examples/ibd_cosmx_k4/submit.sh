#!/bin/bash
#SBATCH --job-name=cell2location_run
#SBATCH --output=/path/to/output_dir/cell2location_run.out
#SBATCH --error=/path/to/output_dir/cell2location_run.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --mem=500gb
#SBATCH --qos=standard
#SBATCH --mail-user=you@example.edu
#SBATCH --mail-type=ALL

module load conda
conda activate /path/to/conda/env
mkdir -p /path/to/output_dir
python runs/ibd_cosmx_k4/ibd_cosmx_k4_pipeline.py
