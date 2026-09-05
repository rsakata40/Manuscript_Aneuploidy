#!/bin/bash -l
#SBATCH --job-name=xcltk_batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/notebooks/7_xclone/0_xcltk_batch/log/%j.xcltk_batch.log
#SBATCH --error=/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/notebooks/7_xclone/0_xcltk_batch/log/%j.xcltk_batch.err
#SBATCH --mail-type=END
#SBATCH --mail-user=rsakata@mrc-lmb.cam.ac.uk

# ---------------------------------------------------------------------------
# Run with:
#   sbatch /ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/notebooks/7_xclone/0_xcltk_batch/0_xcltk_batch.sh
# ---------------------------------------------------------------------------

SAMPLE_LIST="/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/notebooks/7_xclone/0_xcltk_batch/samples.txt"
SCRDIR="/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/notebooks/7_xclone"

mkdir -p \
  "/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/notebooks/7_xclone/0_xcltk_batch/log" \
  "${SCRDIR}/1_xcltk_rdr" \
  "${SCRDIR}/2_xcltk_baf"

while IFS= read -r S; do
  # skip blanks/comments
  [[ -z "${S}" || "${S}" =~ ^# ]] && continue

  echo "Submitting jobs for Sample: ${S}"

  # --- Stage 1: RDR ---
  # We use --parsable to get just the number (e.g., 12345)
  # J1=$(sbatch --parsable \
  #   --job-name="xcltk_rdr_${S}" \
  #   --cpus-per-task=10 \
  #   --mem=100G \
  #   --output="${SCRDIR}/1_xcltk_rdr/log/%j.${S}.1_rdr.log" \
  #   --error="${SCRDIR}/1_xcltk_rdr/log/%j.${S}.1_rdr.err" \
  #   --mail-type=END \
  #   --mail-user=rsakata@mrc-lmb.cam.ac.uk \
  #   --export=ALL,SAMPLE="${S}" \
  #   --wrap="bash '${SCRDIR}/1_xcltk_rdr/1_xcltk_rdr.sh'")

  # --- Stage 2: BAF ---
  # (Runs in parallel with J1 unless you add dependency)
  J2=$(sbatch --parsable \
    --job-name="xcltk_baf_${S}" \
    --cpus-per-task=10 \
    --mem=100G \
    --output="${SCRDIR}/2_xcltk_baf/log/%j.${S}.2_baf.log" \
    --error="${SCRDIR}/2_xcltk_baf/log/%j.${S}.2_baf.err" \
    --mail-type=END \
    --mail-user=rsakata@mrc-lmb.cam.ac.uk \
    --export=ALL,SAMPLE="${S}" \
    --wrap="bash '${SCRDIR}/2_xcltk_baf/2_xcltk_baf.sh'")

done < "$SAMPLE_LIST"