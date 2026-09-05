# count_only_baf.py

from xcltk.baf.fc.main import afc_wrapper
import os, sys, argparse

p = argparse.ArgumentParser(add_help=False)
p.add_argument("--sample", "-s", default=os.environ.get("SAMPLE"))
args = p.parse_args()
SAMPLE = args.sample

OUTDIR    = "/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/processed_data/7_xclone/2_2_xcltk_baf_prephased/" + SAMPLE
BAM="/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/data/cellranger900_count_49831_"+SAMPLE+"_GRCh38-2024-A/possorted_genome_bam.bam"
BARCODES="/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/data/cellranger900_count_49831_"+SAMPLE+"_GRCh38-2024-A/filtered_feature_bc_matrix/barcodes.tsv.gz"
REGION="/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/processed_data/7_xclone/others/annotate_genes_hg38_2024A_xclone.txt" 

#PHASED= "/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/data/for_xclone/deepvariant.hifi_phased.vcf.gz"
PHASED = "/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/processed_data/7_xclone/2_2_xcltk_baf_prephased/"+ SAMPLE +"/phased_filtered.vcf.gz" #filtered vcf
CELL_SNP  = "/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/processed_data/7_xclone/2_xcltk_baf/" + SAMPLE + '/1_pileup'

afc_wrapper(
    sam_fn=BAM,
    barcode_fn=BARCODES,
    region_fn=REGION,
    phased_snp_fn=PHASED,
    out_dir=OUTDIR + "/3_baf_fc",
    cellsnp_dir=CELL_SNP,       
    ncores=10,
    cell_tag="CB", umi_tag="UB",
    min_count=1, min_maf=0,
    output_all_reg=True, no_dup_hap=True,
    min_mapq=20, min_len=30, incl_flag=0, excl_flag=None, no_orphan=True
)
