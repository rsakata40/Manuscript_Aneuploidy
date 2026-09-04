#!/bin/bash
#BSUB -G team292
#BSUB -q normal
#BSUB -n 32
#BSUB -M 100000
#BSUB -R "select[mem>100000] rusage[mem=100000] span[hosts=1]"
#BSUB -J pbmm2
#BSUB -o /nfs/team292/rs40/projects/Aneuploid_screen/notebooks/pacbio/log/%J.pbmm2.log
#BSUB -e /nfs/team292/rs40/projects/Aneuploid_screen/notebooks/pacbio/log/%J.pbmm2.err

set -euo pipefail

cd /nfs/team292/rs40/projects/Aneuploid_screen/notebooks/pacbio

conda activate /nfs/team292/rs40/my-conda-envs/pacbio

pbmm2 align --sort --preset "HIFI" --log-level INFO /lustre/scratch126/cellgen/vento/rs40/data/ref/Homo_sapiens.GRCh38.dna.primary_assembly.fa /lustre/scratch126/cellgen/vento/rs40/pacbio/m84047_240913_121119_s2.hifi_reads.bc2073.bam /lustre/scratch126/cellgen/vento/rs40/pacbio/pbmm2/m84047_240913_121119_s2.hifi_reads.bc2073.aligned.bam