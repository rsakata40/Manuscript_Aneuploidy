#!/bin/bash
source ~/.bashrc
conda activate xclone

SAMPLE="${SAMPLE:-${1:?Usage: $0 SAMPLE}}"
echo "[INFO] SAMPLE=${SAMPLE}"

THREADS=10

# --- paths ---
BAM="/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/data/cellranger900_count_49831_${SAMPLE}_GRCh38-2024-A/possorted_genome_bam.bam"
BARCODES="/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/data/cellranger900_count_49831_${SAMPLE}_GRCh38-2024-A/filtered_feature_bc_matrix/barcodes.tsv.gz"
REGION="/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/processed_data/7_xclone/others/annotate_genes_hg38_2024A_xclone.txt" #from nfs/projects/Aneuploid_screen_v2/notebooks/7_xclone/create_annotate_genes_hg38_2024A.py
OUT_BASE="/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/processed_data/7_xclone/1_xcltk_rdr"
OUTDIR="${OUT_BASE}/${SAMPLE}"

mkdir -p $OUTDIR

xcltk basefc     \
    --sam          "${BAM}"      \
    --barcode      "${BARCODES}"  \
    --region       "${REGION}"   \
    --outdir       "${OUTDIR}"     \
    --ncores       10

echo "[DONE] Outputs in: ${OUTDIR}"
