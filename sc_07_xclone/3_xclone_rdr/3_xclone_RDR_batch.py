import xclone
import pandas as pd
import os
import anndata as ad

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

RDR_outdir = "/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/processed_data/7_xclone/3_xclone_rdr"

os.makedirs(RDR_outdir, exist_ok=True)

#load xcltk for all samples as adatas
data_dir = {}
RDR_adata = {} 
RDR_file = {}
mtx_barcodes_file = {}
regions_anno_file = {}

hg38_genes = xclone.pp.load_anno(genome_mode = "hg38_genes")
hg38_blocks = xclone.pp.load_anno(genome_mode = "hg38_blocks")

for sample in sample_list:
    data_dir[sample] =  "/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/processed_data/7_xclone/1_xcltk_rdr/" + sample +"/"
    RDR_file[sample] = data_dir[sample] + "matrix.mtx"
    mtx_barcodes_file[sample] = data_dir[sample] + "barcodes.tsv"
    regions_anno_file[sample] = data_dir[sample] + "features.tsv" # feature annnotation

    # use default gene annotation
    RDR_adata[sample] = xclone.pp.xclonedata(RDR_file[sample],
                     data_mode = 'RDR',
                     mtx_barcodes_file = mtx_barcodes_file[sample],
                     #regions_anno_file = regions_anno_file[sample],
                     genome_mode = "hg38_genes",
                     data_notes = None)

    
#load annotation file    
anno_df = pd.read_csv('/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/processed_data/3_scanpy_integration/meta_integrated_sub.csv')
anno_df_sub = anno_df[anno_df["dataset"].isin(sample_list)].copy()
anno_df_sub['barcodes_key'] = anno_df_sub['cell.ID_']
anno_df_sub.to_csv(RDR_outdir + "/subset_meta.csv", index=False)

#concat the adatas
adata_merged = ad.concat(
    RDR_adata,
    axis=0,               
    join="outer",               
    index_unique="_",      
    fill_value=0,          
    merge="same"           
)

# keep only cells that appear in subset_meta.csv
keep_cells = adata_merged.obs_names.isin(anno_df_sub["barcodes_key"].values)
adata_merged = adata_merged[keep_cells].copy()

#add annotations
RDR_adata = xclone.pp.extra_anno(adata_merged, RDR_outdir+"/subset_meta.csv", barcodes_key = "barcodes_key",
            cell_anno_key =  ["dataset", "sample", "celltype_coarse"], sep = ",")


#configs
xconfig = xclone.XCloneConfig(dataset_name = "batch", module = "RDR")
xconfig.outdir = RDR_outdir
xconfig.xclone_plot= True

##reference cells
xconfig.cell_anno_key = "dataset"
xconfig.multi_refcelltype = True
xconfig.ref_celltype = ["Embryo-Imp15265938",  "Embryo-Imp15265940", "Embryo-Imp15265944"]
xconfig.select_normal_chr_num = 10

##For removing top marker genes per celltype
xconfig.marker_group_anno_key = "celltype_coarse"
xconfig.get_marker_genes = True
xconfig.top_n_marker = 50
xconfig.exclude_XY=True
xconfig.remove_guide_XY=True

#denoise
xconfig.RDR_denoise = False
#xconfig.n_neighbors = 0

#plot settings
xconfig.set_figure_params(xclone= True, fontsize = 18)
xconfig.plot_cell_anno_key =  'sample'

xconfig.display()

#run xclone RDR
RDR_merge_Xdata = xclone.model.run_RDR(RDR_adata,
            config_file = xconfig)

#save
RDR_merge_Xdata.obs['scrublet_prediction'] = RDR_merge_Xdata.obs['scrublet_prediction'].astype(str)
RDR_merge_Xdata.write(RDR_outdir + "/RDR_adata_KNN_HMM_post.h5ad")
