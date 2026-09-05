#!/bin/bash -l
#SBATCH --job-name=xclone_combine
#SBATCH --cpus-per-task=20                     
#SBATCH --mem=750G                
#SBATCH --output=/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/notebooks/7_xclone/5_2_xclone_combine_prephased/log/%x-%j.out
#SBATCH --error=/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/notebooks/7_xclone/5_2_xclone_combine_prephased/log/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=rsakata@mrc-lmb.cam.ac.uk

# --- Environment -------------------------------------------------------------
#run sbatch /ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/notebooks/7_xclone/5_2_xclone_combine_prephased/5_2_xclone_combine.sh
conda activate xclone
cd /ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/notebooks/7_xclone/5_2_xclone_combine_prephased/log

# Match thread usage to allocated CPUs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "Job $SLURM_JOB_ID on $(hostname)"

python -V && which python

# --- Run ---------------------------------------------------------------------

/usr/bin/time -v python /ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/notebooks/7_xclone/5_2_xclone_combine_prephased/5_2_clone_combine_celltype.py

