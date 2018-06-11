#!/bin/sh
#SBATCH --job-name="personality-detection-cnn"
#SBATCH --output=result_cnn.out
#SBATCH --mail-user=marahimi@ucsc.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=128x24
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory="$SLURM_SUBMIT_DIR
export OMPI_MCA_btl=tcp,sm,sel
srun python3 personality_cnn.py
