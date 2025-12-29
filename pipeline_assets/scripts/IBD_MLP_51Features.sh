#!/bin/bash
#SBATCH --job-name=IBD_MLP_51Features
#SBATCH --output=/blue/pbenos/tan.m/IBDCosMx_scRNAseq/CosMx/Post-NMF_Analysis/MLP_51Features/IBD_MLP.out
#SBATCH --error=/blue/pbenos/tan.m/IBDCosMx_scRNAseq/CosMx/Post-NMF_Analysis/MLP_51Features/IBD_MLP.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32  # Adjust based on the number of CPUs you want to use
#SBATCH --nodes=1           # Number of nodes
#SBATCH --time=72:00:00     # Max time (hh:mm:ss)
#SBATCH --mem=500gb           # Memory per node
#SBATCH --qos=pbenos-b   # Partition to submit to
#SBATCH --mail-user=tan.m@ufl.edu
#SBATCH --mail-type=ALL


# Load the modules
module load conda

# Activate the Conda environment
conda activate /blue/pbenos/tan.m/cell2location_cuda118_torch22

echo "Running MLP_51Features classification on IBD data"

# Run the R script
python /blue/pbenos/tan.m/IBDCosMx_scRNAseq/IBD_MLP_51Features.py