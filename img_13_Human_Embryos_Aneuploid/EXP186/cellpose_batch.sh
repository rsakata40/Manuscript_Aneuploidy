#!/bin/bash -l
#SBATCH --job-name=186_cellpose
#SBATCH --partition=ml            # gpu | agpu (RTX4090) | ml (A100, ML only)
#SBATCH --gres=gpu:4     # <-- change to 1–8 GPUs on 'gpu' partition (types: gtx1080ti, rtx2080ti)
# For 'agpu' (RTX4090 only), instead use:   #SBATCH --gres=gpu:1   (up to 4 per node)
# For 'ml'  (A100-40/80 only), e.g.:        #SBATCH --gres=gpu:8
 ## SBATCH --time=48:00:00             # Max allowed is 7-00:00:00
#SBATCH --output=/ceph.groups/mshahbazi.grp/rsakata/HE_EXP179/cellpose/code/log/%x-%j.out
#SBATCH --error=/ceph.groups/mshahbazi.grp/rsakata/HE_EXP179/cellpose/code/log/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=rsakata@mrc-lmb.cam.ac.uk

#SBATCH --array=0-12%4 #number of files-1 % GPU allowed

# --- Environment -------------------------------------------------------------
#run by sbatch /ceph.groups/mshahbazi.grp/rsakata/HE_EXP186/cellpose/code/cellpose_batch.sh
conda activate cellpose_env

# Good practice: match thread counts to allocated CPUs (Slurm sets 8 CPUs per GPU on 'gpu' by default)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "Job $SLURM_JOB_ID on $(hostname)"

python -V && which python
python - <<'PY'
import torch, os
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA runtime:", torch.version.cuda)
print("Visible GPUs:", os.getenv("CUDA_VISIBLE_DEVICES"))
PY

# Print GPUs if available; otherwise say so, without failing
nvidia-smi || echo "nvidia-smi not available"

# --- Run ---------------------------------------------------------------------

# Pass any extra flags you need to your script after this file name:
python /ceph.groups/mshahbazi.grp/rsakata/HE_EXP186/cellpose/code/cellpose_batch.py
