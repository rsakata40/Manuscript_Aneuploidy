#!/bin/bash -l
#SBATCH --job-name=rdr_batch
#SBATCH --cpus-per-task=24                     
#SBATCH --mem=700G                
#SBATCH --output=/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/notebooks/7_xclone/3_xclone_rdr/log/%x-%j.out
#SBATCH --error=/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/notebooks/7_xclone/3_xclone_rdr/log/%x-%j.err
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

python /ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/notebooks/7_xclone/3_xclone_rdr/3_xclone_RDR_batch.py

# --- Resource Summary (LSF-style equivalent) ---------------------------------
echo "----------------------------------------------------------------"
echo "Resource usage summary for Job $SLURM_JOB_ID:"
echo "----------------------------------------------------------------"

seff $SLURM_JOB_ID

echo "----------------------------------------------------------------"