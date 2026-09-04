#!/bin/bash

CPU=10
MEM=100000   # MB

LOGDIR=/nfs/team292/rs40/projects/Aneuploid_screen_v2/notebooks/4_scploid/log
SCRIPT=/nfs/team292/rs40/projects/Aneuploid_screen_v2/notebooks/4_scploid/scploid.R
ENV=R_scploid

mkdir -p "$LOGDIR"

cd /nfs/team292/rs40/projects/Aneuploid_screen_v2/notebooks/4_scploid

bsub \
  -G team292 \
  -n "$CPU" \
  -M "$MEM" \
  -R "select[mem>$MEM] rusage[mem=$MEM]" \
  -o "$LOGDIR/scploid_%J.log" \
  -e "$LOGDIR/scploid_%J.error" \
  bash -lc "conda activate $ENV; R CMD BATCH $SCRIPT"

