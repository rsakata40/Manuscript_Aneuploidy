#!/bin/bash
#BSUB -G team292
#BSUB -q normal
#BSUB -n 16
#BSUB -M 10000
#BSUB -R "select[mem>10000] rusage[mem=10000] span[hosts=1]"
#BSUB -J hiphase
#BSUB -o /nfs/team292/rs40/projects/Aneuploid_screen/notebooks/pacbio/log/%J.hiphase.log
#BSUB -e /nfs/team292/rs40/projects/Aneuploid_screen/notebooks/pacbio/log/%J.hiphase.err

set -euo pipefail

# 1. Environment Setup
module load cellgen/singularity
module load cellgen/samtools
conda activate hiphase

# Create log dir if missing
mkdir -p /nfs/team292/rs40/projects/Aneuploid_screen/notebooks/pacbio/log

# --- Configuration ---
WD="/lustre/scratch126/cellgen/vento/rs40/pacbio"
IN_BAM="/lustre/scratch126/cellgen/vento/rs40/pacbio/pbmm2/m84047_240913_121119_s2.hifi_reads.bc2073.aligned.bam"

IN_VCF="${WD}/deepvariant/deepvariant_output.vcf.gz"
OUT_VCF="${WD}/4_hiphase/deepvariant.hifi_phased.vcf.gz"
REFERENCE="/lustre/scratch126/cellgen/vento/rs40/data/ref/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
THREADS="16"

# Output Directory
OUTDIR="${WD}/4_hiphase"
mkdir -p "${OUTDIR}"

# Run
echo "Running HiPhase..."

hiphase \
    --bam ${IN_BAM} \
    --vcf ${IN_VCF} \
    --output-vcf ${OUT_VCF} \
    --reference ${REFERENCE} \
    --threads ${THREADS}

# --- Indexing ---
tabix -p vcf "$OUT_VCF"

echo "Done."