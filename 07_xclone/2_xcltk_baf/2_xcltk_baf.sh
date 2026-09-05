#!/bin/bash
source ~/.bashrc
conda activate xclone

# --- paths ---
SAMPLE="${SAMPLE:-${1:?Usage: $0 SAMPLE}}"
echo "[INFO] SAMPLE=${SAMPLE}"

OUT_BASE="/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/processed_data/7_xclone/2_xcltk_baf"
OUTDIR="${OUT_BASE}/${SAMPLE}"

BAM="/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/data/cellranger900_count_49831_${SAMPLE}_GRCh38-2024-A/possorted_genome_bam.bam"
BARCODES="/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/data/cellranger900_count_49831_${SAMPLE}_GRCh38-2024-A/filtered_feature_bc_matrix/barcodes.tsv.gz"
THREADS=10

VCF="/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/data/for_xclone/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz"
REGION="/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/processed_data/7_xclone/others/annotate_genes_hg38_2024A_xclone.txt" 

GMAP="/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/data/for_xclone/Eagle_v2.4.1/tables/genetic_map_hg38_withX.txt.gz"
EAGLE="/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/data/for_xclone/Eagle_v2.4.1/eagle"
PANEL="/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/data/for_xclone/1000G_hg38"
GENOME="hg38"

mkdir -p "$OUTDIR" 

xcltk baf \
  --label   "$SAMPLE" \
  --sam     "$BAM" \
  --barcode "$BARCODES" \
  --snpvcf  "$VCF" \
  --region  "$REGION" \
  --outdir  "$OUTDIR" \
  --gmap    "$GMAP" \
  --eagle   "$EAGLE" \
  --paneldir "$PANEL" \
  --ncores  "$THREADS"

echo "[DONE] Outputs in: ${OUTDIR}"

