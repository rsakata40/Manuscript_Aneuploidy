#!/bin/bash -l
#SBATCH --job-name=baf_batch
#SBATCH --cpus-per-task=20                     
#SBATCH --mem=100G                
#SBATCH --output=/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/notebooks/7_xclone/4_2_xclone_baf_prephased/log/%x-%j.out
#SBATCH --error=/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/notebooks/7_xclone/4_2_xclone_baf_prephased/log/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=rsakata@mrc-lmb.cam.ac.uk

# --- Environment -------------------------------------------------------------
#run 
conda activate xclone

# Match thread usage to allocated CPUs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "Job $SLURM_JOB_ID on $(hostname)"

python -V && which python

# --- Run ---------------------------------------------------------------------

python /ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/notebooks/7_xclone/4_2_xclone_baf_prephased/4_xclone_BAF_batch.py

# --- Resource Summary (LSF-style equivalent) ---------------------------------
echo "----------------------------------------------------------------"
echo "Resource usage summary for Job $SLURM_JOB_ID:"
echo "----------------------------------------------------------------"

# 'seff' prints CPU efficiency, Memory usage, and wall-clock time
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,MaxRSS,Elapsed,State

echo "----------------------------------------------------------------"