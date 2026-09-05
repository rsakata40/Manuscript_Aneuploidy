#!/bin/bash -l
#SBATCH --job-name=xcltk_batch_pre
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/notebooks/7_xclone/2_2_xcltk_baf_prephased/log/%j.xcltk_baf_preprocessed.log
#SBATCH --error=/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/notebooks/7_xclone/2_2_xcltk_baf_prephased/log/%j.xcltk_baf_preprocessed.err
#SBATCH --mail-type=END
#SBATCH --mail-user=rsakata@mrc-lmb.cam.ac.uk

# ---------------------------------------------------------------------------
# Run with:
#   sbatch /ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/notebooks/7_xclone/2_2_xcltk_baf_prephased/2_2_xcltk_baf_prephased.sh
# ---------------------------------------------------------------------------

SAMPLE_LIST="/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/notebooks/7_xclone/0_xcltk_batch/samples.txt"
OUT_DIR="/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/processed_data/7_xclone/2_2_xcltk_baf_prephased"
LOG_DIR="/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/notebooks/7_xclone/2_2_xcltk_baf_prephased/log"

mkdir -p \
  "${LOG_DIR}" \
  "${OUT_DIR}"

while IFS= read -r S; do
  # skip blanks/comments
  [[ -z "${S}" || "${S}" =~ ^# ]] && continue

  echo "Submitting jobs for Sample: ${S}"
  
  # Define the directory path for this sample's data
  SAMPLE_OUT_DIR="/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/processed_data/7_xclone/2_2_xcltk_baf_prephased/${S}"
  
  mkdir -p "${SAMPLE_OUT_DIR}/3_baf_fc"

  J1=$(sbatch --parsable \
    --job-name="xcltk_prephase_${S}" \
    --cpus-per-task=10 \
    --mem=200G \
    --output="${LOG_DIR}/%j.${S}.2_1prephase.log" \
    --error="${LOG_DIR}/%j.${S}.2_1prephase.err" \
    --mail-type=END \
    --mail-user=rsakata@mrc-lmb.cam.ac.uk \
    --export=ALL,SAMPLE="${S}" \
    --wrap="
          source ~/.bashrc
          conda activate xclone
          
          PHASED_VCF='/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/data/for_xclone/deepvariant.hifi_phased.vcf.gz'
          CELLSNP_VCF='/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/processed_data/7_xclone/2_xcltk_baf/${S}/1_pileup/cellSNP.base.vcf.gz'
          OUTPUT_VCF='${SAMPLE_OUT_DIR}/phased_filtered.vcf.gz'
          
          # 1. Sort the CellSNP VCF (Fixes 'Unsorted positions' error)
          echo 'Sorting CellSNP VCF for ${S}...'
          bcftools sort \"\$CELLSNP_VCF\" -Oz -o \"\${CELLSNP_VCF}.tmp.gz\"
          mv \"\${CELLSNP_VCF}.tmp.gz\" \"\$CELLSNP_VCF\"
          
          # 2. Index the sorted file
          bcftools index -t \"\$CELLSNP_VCF\"

          # 3. Filter
          echo 'Filtering VCF for ${S}...'
          bcftools isec -n 2 -w 1 \"\$PHASED_VCF\" \"\$CELLSNP_VCF\" -Oz -o \"\$OUTPUT_VCF\"
          
          # 4. Index output
          bcftools index -t \"\$OUTPUT_VCF\"
          
          # 5. Run Python
          python /ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/notebooks/7_xclone/2_2_xcltk_baf_prephased/2_2_xcltk_baf_prephased.py
        ")

done < "$SAMPLE_LIST"