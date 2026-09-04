#!/bin/bash
#BSUB -G team292
#BSUB -q normal
#BSUB -n 32
#BSUB -M 64000
#BSUB -R "select[mem>64000] rusage[mem=64000] span[hosts=1]"
#BSUB -J deepvariant
#BSUB -o /nfs/team292/rs40/projects/Aneuploid_screen/notebooks/pacbio/log/%J.deepvariant.log
#BSUB -e /nfs/team292/rs40/projects/Aneuploid_screen/notebooks/pacbio/log/%J.deepvariant.err

set -euo pipefail

# 1. Source Conda properly before activating
module load cellgen/conda
conda activate deepvariant_whatshap

module load cellgen/singularity
module load cellgen/samtools

mkdir -p /nfs/team292/rs40/projects/Aneuploid_screen/notebooks/pacbio/log
cd /nfs/team292/rs40/projects/Aneuploid_screen/notebooks/pacbio


# --- Configuration ---
WD="/lustre/scratch126/cellgen/vento/rs40/pacbio"
REF="/lustre/scratch126/cellgen/vento/rs40/data/ref/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
BAM="${WD}/pbmm2/m84047_240913_121119_s2.hifi_reads.bc2073.aligned.bam"
OUTDIR="${WD}/deepvariant"
THREADS=${LSB_DJOB_NUMPROC:-32}

# downloaded 
IMG="/nfs/team292/rs40/projects/Aneuploid_screen/notebooks/pacbio/deepvariant_1.9.0.sif"

export TMPDIR="${OUTDIR}/singularity_tmp"
mkdir -p "$TMPDIR"

# 2. Tell Singularity to use this folder
export SINGULARITYENV_TMPDIR="$TMPDIR"
export SINGULARITYENV_TEMP="$TMPDIR"
export SINGULARITYENV_TMP="$TMPDIR"

echo "Using temp dir: $TMPDIR"

# 3. Run DeepVariant
# We bind the new TMPDIR path implicitly because it is inside /lustre, which is already bound.
singularity exec \
  -B /nfs/team292/rs40,/lustre/scratch126/cellgen/vento/rs40 \
  -B /usr/lib/locale/ \
  "$IMG" \
  /opt/deepvariant/bin/run_deepvariant \
    --model_type PACBIO \
    --ref "$REF" \
    --reads "$BAM" \
    --output_vcf "${OUTDIR}/deepvariant_output.vcf.gz" \
    --output_gvcf "${OUTDIR}/deepvariant_output.g.vcf.gz" \
    --num_shards "$THREADS" \
    --regions "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X Y" \
    --intermediate_results_dir "${OUTDIR}/intermediate_results" \
    --make_examples_extra_args "small_model_call_multiallelics=false"