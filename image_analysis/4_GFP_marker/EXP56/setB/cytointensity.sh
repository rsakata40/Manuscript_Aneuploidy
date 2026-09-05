#!/bin/bash -l
#SBATCH --job-name=cyint56B
#SBATCH --partition=ml                  # gpu | agpu (RTX4090) | ml (A100)
#SBATCH --gres=gpu:1                    # set to 1–4 on agpu, 1–8 on gpu, 1–8 on ml (depending on cluster policy)
#SBATCH --cpus-per-task=8               # CPU workers for I/O & preprocessing
#SBATCH --time=48:00:00                 # Max allowed often 7-00:00:00
#SBATCH --mem=100G
#SBATCH --output=/ceph.groups/mshahbazi.grp/rsakata/EXP56/cellpose/code/setB/output/1320-%x-%j.out
#SBATCH --error=/ceph.groups/mshahbazi.grp/rsakata/EXP56/cellpose/code/setB/output/1320-%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rsakata@mrc-lmb.cam.ac.uk

# Optional: run as an array (one file per task). Uncomment and set range:
#SBATCH --array=0-39%4 #number of files-1 % GPU allowed

# --- Environment -------------------------------------------------------------
# Usage:
#   sbatch /ceph.groups/mshahbazi.grp/rsakata/EXP56/cellpose/code/setB/cytointensity.sh
# Before submitting, ensure the conda env exists and includes: numpy, scipy, scikit-image, tifffile, matplotlib, natsort, pandas

conda activate cellpose_env

# Thread sanity: match to CPUs and avoid oversubscription by BLAS/OpenMP
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

# (Optional) deterministic I/O sorting etc
export LC_ALL=C

cd /ceph.groups/mshahbazi.grp/rsakata/EXP56/cellpose/code/setB

echo "Job $SLURM_JOB_ID on $(hostname)"
date

python -V && which python
python - <<'PY'
import torch, os
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA runtime:", getattr(torch.version, "cuda", None))
print("Visible GPUs:", os.getenv("CUDA_VISIBLE_DEVICES"))
PY

# Print GPUs if available; otherwise say so, without failing
nvidia-smi || echo "nvidia-smi not available"

# --- Run ---------------------------------------------------------------------
# /usr/bin/time -v captures peak memory (MaxRSS) and CPU stats in stdout
/usr/bin/time -v \
python /ceph.groups/mshahbazi.grp/rsakata/EXP56/cellpose/code/setB/cytointensity.py

# If you need to pass custom args to your script, append them after the .py above.
# Example:
# /usr/bin/time -v python .../structureseg_batch.py --method otsu --do-qc
