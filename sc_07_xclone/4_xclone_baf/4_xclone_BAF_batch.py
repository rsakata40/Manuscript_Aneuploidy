import xclone
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import anndata as ad
import os

BAF_outdir = "/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/processed_data/7_xclone/4_xclone_baf"
os.makedirs(BAF_outdir, exist_ok=True)

sample_list = ['Embryo-Imp15265938', 
               'Embryo-Imp15265939', 
               'Embryo-Imp15265940',
               'Embryo-Imp15265941',
               'Embryo-Imp15265942',
               'Embryo-Imp15265943',
               'Embryo-Imp15265944',
               'Embryo-Imp15265945',
               'Embryo-Imp15265946',
               'Embryo-Imp15265947']

#load xcltk for all samples as adatas
data_dir = {}
AD_file = {}
DP_file = {}
mtx_barcodes_file = {}
BAF_adata = {}

for sample in sample_list:
    data_dir[sample] = "/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/processed_data/7_xclone/2_xcltk_baf/" +sample + "/3_baf_fc/"
    AD_file[sample] = data_dir[sample] + "xcltk.AD.mtx"
    DP_file[sample] = data_dir[sample] + "xcltk.DP.mtx"
    mtx_barcodes_file[sample] = data_dir[sample] + "xcltk.samples.tsv" # cell barcodes

    # use default gene annotation
    BAF_adata[sample] = xclone.pp.xclonedata([AD_file[sample], DP_file[sample]], 'BAF',
                                     mtx_barcodes_file[sample],
                                     genome_mode = "hg38_genes")

    
#load annotation file    
anno_file = '/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/processed_data/3_scanpy_integration/meta_integrated_sub.csv'
anno_df = pd.read_csv(anno_file)
anno_df_sub = anno_df[anno_df["dataset"].isin(sample_list)].copy()
anno_df_sub['barcodes_key'] = anno_df_sub['cell.ID_']
anno_df_sub.to_csv(BAF_outdir+"/subset_meta.csv", index=False)


#concat the adatas
adata_merged = ad.concat(
    BAF_adata,
    axis=0,                # stack cells (rows)
    join="outer",          # union of genes (missing filled with 0/NaN)
    #label="batch",         # new obs column with sample ID
    #keys=samples,          # values for that column
    index_unique="_",      # make unique cell IDs like "<old>-<batch>"
    fill_value=0,          # for missing genes/layers
    merge="same"           # only keep .uns/etc if identical across batches
)

# keep only cells that appear in subset_meta.csv
keep_cells = adata_merged.obs_names.isin(anno_df_sub["barcodes_key"].values)
adata_merged = adata_merged[keep_cells].copy()

#add annotations
BAF_adata = xclone.pp.extra_anno(adata_merged, BAF_outdir+"/subset_meta.csv", barcodes_key = "barcodes_key",
            cell_anno_key = ["cell.ID_", "dataset", "celltype_coarse",  "sample"], sep = ",")


#assign control samples as ref cell
BAF_adata.obs["ref_cell"] = np.where(BAF_adata.obs[ "dataset"].isin(["Embryo-Imp15265938",  "Embryo-Imp15265940", "Embryo-Imp15265944"]), 'Y', 'N')
BAF_adata.obs["ref_cell"] = BAF_adata.obs["ref_cell"].astype('category')


#configs
xconfig = xclone.XCloneConfig(dataset_name = "merged", module = "BAF")
xconfig.update_info_from_rdr= False
xconfig.outdir = BAF_outdir

## reference
xconfig.cell_anno_key = "ref_cell"
xconfig.ref_celltype = "Y"
xconfig.select_normal_chr_num = 10

## subset
xconfig.exclude_XY=True
xconfig.remove_guide_XY=True

## denoise
xconfig.BAF_denoise = False

## plots
xconfig.xclone_plot= True
xconfig.plot_cell_anno_key = "dataset"
xconfig.set_figure_params(xclone= True, fontsize = 18)
xconfig.plot_remove_reference =  False

xconfig.display()

#run xclone BAF
BAF_merge_Xdata = xclone.model.run_BAF(BAF_adata,
            config_file = xconfig)


#save
BAF_merge_Xdata.obs['scrublet_prediction'] = BAF_merge_Xdata.obs['scrublet_prediction'].astype(str)
BAF_merge_Xdata.write(BAF_outdir + "/BAF_merge_Xdata_KNN_HMM_post.h5ad")