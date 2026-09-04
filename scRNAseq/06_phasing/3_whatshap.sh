#!/bin/bash
#BSUB -G team292
#BSUB -q normal
#BSUB -n 8                     
#BSUB -M 20000
#BSUB -R "select[mem>20000] rusage[mem=20000] span[hosts=1]"
#BSUB -J whatshap
#BSUB -o /nfs/team292/rs40/projects/Aneuploid_screen/notebooks/pacbio/log/whatshap.%J.log
#BSUB -e /nfs/team292/rs40/projects/Aneuploid_screen/notebooks/pacbio/log/whatshap.%J.err


set -euo pipefail

# 1. Environment Setup
module load cellgen/conda
conda activate deepvariant_whatshap  
module load cellgen/samtools


# --- Configuration ---
/lustre/scratch126/cellgen/vento/rs40/pacbio/pbmm2/m84047_240913_121119_s2.hifi_reads.bc2073.aligned.bam
WD="/lustre/scratch126/cellgen/vento/rs40/pacbio"
REF="/lustre/scratch126/cellgen/vento/rs40/data/ref/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
BAM="${WD}/pbmm2/m84047_240913_121119_s2.hifi_reads.bc2073.aligned.bam"

# Input VCF (From DeepVariant)
VCF="${WD}/deepvariant/deepvariant_output.vcf.gz"

# Output VCF (Phased)
OUTDIR="${WD}/3_whatshap"
OUT_PHASED="${OUTDIR}/H9_phased_whatshap.vcf.gz"

mkdir -p "${OUTDIR}"

# 3. Run Whatshap
echo "Starting Whatshap phasing..."

whatshap phase \
    --reference "$REF" \
    --output "$OUT_PHASED" \
    --ignore-read-groups \
    --indels \
    "$VCF" \
    "$BAM"

# 4. Index the result
echo "Indexing phased VCF..."
tabix -p vcf "$OUT_PHASED"

echo "Done."