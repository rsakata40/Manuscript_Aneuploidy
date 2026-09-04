bsub \
  -J scrublet_batch \
  -G team292 \
  -q normal \
  -n 8 \
  -M 50000 \
  -R 'select[mem>50000] rusage[mem=50000] span[hosts=1]' \
  -o /nfs/team292/rs40/projects/Aneuploid_screen_v2/notebooks/1_scrublet/log/%J.log \
  -e /nfs/team292/rs40/projects/Aneuploid_screen_v2/notebooks/1_scrublet/log/%J.err \
  'conda activate py_scanpy; python /nfs/team292/rs40/projects/Aneuploid_screen_v2/notebooks/1_scrublet/scrublet_batch.py'
  