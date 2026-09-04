#!/bin/bash
#BSUB -G team292
#BSUB -q normal
#BSUB -n 8                     
#BSUB -M 8000
#BSUB -R "select[mem>8000] rusage[mem=8000] span[hosts=1]"
#BSUB -J bam_index
#BSUB -o /nfs/team292/rs40/projects/Aneuploid_screen/notebooks/pacbio/log/bam_index.%J.log
#BSUB -e /nfs/team292/rs40/projects/Aneuploid_screen/notebooks/pacbio/log/bam_index.%J.err

# 1. Load samtools (Use the method that works on your cluster)
# method A: module load
module load cellgen/samtools

# method B: conda (uncomment if you use conda)
# source activate my_env

# 2. Run the command with multi-threading
samtools index -@ 8 /lustre/scratch126/cellgen/vento/rs40/pacbio/m84047_240913_121119_s2.hifi_reads.bc2073.bam

echo "Indexing complete."