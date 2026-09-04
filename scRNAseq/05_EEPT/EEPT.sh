#!/bin/bash
#BSUB -J EEPT           # Job Name
#BSUB -q normal               # Queue
#BSUB -G team292
#BSUB -n 10                    # Cores
#BSUB -M 100G                
#BSUB -R "select[mem>100G ] rusage[mem=100G ] span[hosts=1]"
#BSUB -o /nfs/team292/rs40/projects/Aneuploid_screen_v2/notebooks/6_EEPT/log/%J_eept.out          # Output log
#BSUB -e /nfs/team292/rs40/projects/Aneuploid_screen_v2/notebooks/6_EEPT/log/%J_eept.err          # Error log

# Define paths
SIF_IMAGE="/nfs/team292/rs40/singularity/shiny-tools-eeptools.sif" # CHECK THIS PATH
R_SCRIPT="/nfs/team292/rs40/projects/Aneuploid_screen_v2/notebooks/6_EEPT/EEPT.R" 

module load cellgen/singularity

# Run Singularity
# We bind /nfs so the container can see your data and the R script
singularity exec \
  --bind /nfs:/nfs \
  --bind /nfs/team292/rs40:/nfs/team292/rs40 \
  $SIF_IMAGE \
  Rscript $R_SCRIPT