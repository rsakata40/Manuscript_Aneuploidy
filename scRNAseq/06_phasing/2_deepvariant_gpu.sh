#!/bin/bash
#BSUB -G team292
#BSUB -q gpu-normal
#BSUB -n 1
#BSUB -M 64000
#BSUB -J deepvariant_gpu
#BSUB -o /nfs/team292/rs40/projects/Aneuploid_screen/notebooks/pacbio/log/%J.gpu_deepvariant.log
#BSUB -e /nfs/team292/rs40/projects/Aneuploid_screen/notebooks/pacbio/log/%J.dpu_deepvariant.err

# --- GPU Request---
#BSUB -gpu "mode=exclusive_process"
#BSUB -R "select[ngpus>0 && mem>64000] rusage[ngpus_physical=1.00,mem=64000] span[ptile=1]"


set -euo pipefail

# --- Environment ---
module load cellgen/conda
conda activate deepvariant_whatshap
module load cellgen/singularity
module load cellgen/samtools

# --- Configuration ---
BIN_VERSION="1.6.1" # Ensure this version matches the image you pull
WD="/lustre/scratch126/cellgen/vento/rs40/pacbio"
REF="/lustre/scratch126/cellgen/vento/rs40/data/ref/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
BAM="${WD}/pbmm2/m84047_240913_121119_s2.hifi_reads.bc2073.aligned.bam"
OUTDIR="${WD}/deepvariant_gpu"
THREADS=${LSB_DJOB_NUMPROC:-16}

mkdir -p "${OUTDIR}/log"

# --- 1. Pull the GPU Image (If not already present) ---
# IMPORTANT: This must be the "-gpu" version, not the standard one!
BIN_VERSION="1.9.0"
SIF_IMAGE="${WD}/deepvariant_${BIN_VERSION}_gpu.sif"

export SINGULARITY_CACHEDIR="${WD}/singularity_cache"
mkdir -p "$SINGULARITY_CACHEDIR"
export TMPDIR="${OUTDIR}/singularity_tmp"
mkdir -p "$TMPDIR"
export SINGULARITYENV_TMPDIR="$TMPDIR"
export SINGULARITYENV_TEMP="$TMPDIR"
export SINGULARITYENV_TMP="$TMPDIR"

if [ ! -f "$SIF_IMAGE" ]; then
    echo "Pulling DeepVariant GPU image..."
    singularity pull "$SIF_IMAGE" docker://google/deepvariant:"${BIN_VERSION}-gpu"
fi

echo "Using temp dir: $TMPDIR"

# --- 3. Run DeepVariant ---
# Added '--nv' to enable GPU support
singularity exec --nv \
  -B /nfs,/lustre \
  -B /usr/lib/locale/ \
  "$SIF_IMAGE" \
  /opt/deepvariant/bin/run_deepvariant \
    --model_type PACBIO \
    --ref "$REF" \
    --reads "$BAM" \
    --output_vcf "${OUTDIR}/deepvariant_output.vcf.gz" \
    --output_gvcf "${OUTDIR}/deepvariant_output.g.vcf.gz" \
    --num_shards "$THREADS" \
    --intermediate_results_dir "${OUTDIR}/intermediate_results"  \
    --disable_small_model=true

# --- Indexing ---
tabix -p vcf "${OUTDIR}/deepvariant_output.vcf.gz" || true
tabix -p vcf "${OUTDIR}/deepvariant_output.g.vcf.gz" || true

echo "Done."