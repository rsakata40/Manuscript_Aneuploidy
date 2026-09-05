#### this version consumes too much memory. run per cell type instead. 
import xclone
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import anndata
import os

outdir = "/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/processed_data/7_xclone/5_xclone_combine"
os.makedirs(outdir , exist_ok=True)

RDR_outdir = "/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/processed_data/7_xclone/3_xclone_rdr/data"
RDR_Xdata = anndata.read_h5ad(RDR_outdir + "/RDR_adata_KNN_HMM_post.h5ad")

BAF_outdir = "/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/processed_data/7_xclone/4_xclone_baf/data"
BAF_merge_Xdata = anndata.read_h5ad(BAF_outdir + "/BAF_merge_Xdata_KNN_HMM_post.h5ad")

flag = ~(BAF_merge_Xdata.var["chr"] == "Y")

BAF_merge_Xdata = BAF_merge_Xdata[:, flag]


#run xclone combine
xconfig = xclone.XCloneConfig(dataset_name = "joint", module = "Combine")

xconfig.set_figure_params(xclone= True, fontsize = 18)
xconfig.outdir = outdir

xconfig.guide_chr_anno_key = "chr"
xconfig.xclone_plot= True
xconfig.plot_cell_anno_key = "sample"
xconfig.merge_loss = True
xconfig.merge_loh = True
xconfig.cell_anno_key = "dataset"
xconfig.ref_celltype = ["Embryo-Imp15265938",  "Embryo-Imp15265940", "Embryo-Imp15265944"]
xconfig.WGD_detection = False
xconfig.BAF_denoise = False
xconfig.RDR_denoise  = False
xconfig.display()

combine_Xdata = xclone.model.run_combine(RDR_Xdata,
                BAF_merge_Xdata,
                verbose = True,
                run_verbose = True,
                config_file = xconfig)

#save
combine_Xdata.obs['scrublet_prediction'] = combine_Xdata.obs['scrublet_prediction'].astype(str)
combine_Xdata.write(outdir + "/Combined_merge_Xdata_KNN_HMM_post.h5ad")
