import xclone
import scanpy as sc
import pandas as pd
import numpy as np
import anndata
import os
import gc  # Garbage collector for memory management

# --- Setup Paths ---
base_outdir = "/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/processed_data/7_xclone/5_xclone_combine"
os.makedirs(base_outdir, exist_ok=True)

RDR_outdir = "/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/processed_data/7_xclone/3_xclone_rdr/data"
BAF_outdir = "/ceph.groups/mshahbazi.grp/rsakata/FromSanger/Aneuploid_screen_v2/processed_data/7_xclone/4_xclone_baf/data"

# --- Load Data ---
print("Loading RDR data...")
RDR_Xdata = anndata.read_h5ad(RDR_outdir + "/RDR_adata_KNN_HMM_post.h5ad", backed='r')

print("Loading BAF data...")
BAF_merge_Xdata = anndata.read_h5ad(BAF_outdir + "/BAF_merge_Xdata_KNN_HMM_post.h5ad", backed='r')

# Filter BAF Chromosomes (Remove Y)
gene_flag = ~(BAF_merge_Xdata.var["chr"] == "Y")      # Just save the mask

# --- Identify Cell Types ---
# Check available cell types in the RDR object
# (Assuming 'celltype_coarse' exists in .obs from your previous steps)
cell_type_key = "celltype_coarse" 
unique_cell_types = RDR_Xdata.obs[cell_type_key].unique()

print(f"Found {len(unique_cell_types)} cell types to process: {unique_cell_types}")

# --- Processing Loop ---
for cell_type in unique_cell_types:
    print(f"\n{'='*40}")
    print(f"Processing group: {cell_type}")
    print(f"{'='*40}")

    # 1. Create specific output directory
    # Replace spaces or slashes in cell type name to avoid path errors
    safe_ct_name = str(cell_type).replace("/", "_").replace(" ", "_")
    current_outdir = os.path.join(base_outdir, safe_ct_name)
    os.makedirs(current_outdir, exist_ok=True)

    # 2. Subset the data
    # Find barcodes belonging to this cell type
    cells_of_interest = RDR_Xdata.obs[cell_type_key] == cell_type
    
    # Check if we have enough cells (optional safety)
    n_cells = cells_of_interest.sum()
    if n_cells < 10:
        print(f"Skipping {cell_type}: Too few cells ({n_cells})")
        continue

    # Slice the AnnData objects
    # Note: We rely on RDR and BAF having matching indices (barcodes). 
    # If they might differ, use common intersection first.
    rdr_sub = RDR_Xdata[cells_of_interest]
    
    # Ensure BAF has the same cells (subset by the index of the RDR subset)
    # This handles cases where BAF might have slightly different cell order or counts
    common_cells = rdr_sub.obs_names.intersection(BAF_merge_Xdata.obs_names)
    
    if len(common_cells) != len(rdr_sub):
        print(f"Warning: BAF data missing for some cells in {cell_type}. Intersecting...")
        rdr_sub = rdr_sub[common_cells]
    
    baf_sub = BAF_merge_Xdata[common_cells, gene_flag].to_memory()
    rdr_sub = rdr_sub.to_memory()
    
    print(f"  > Subset contains {rdr_sub.n_obs} cells.")

    print(
    f"  RDR shape: {rdr_sub.shape}, "
    f"BAF shape: {baf_sub.shape}, "
    f"RDR dtype: {rdr_sub.X.dtype}, "
    f"BAF dtype: {baf_sub.X.dtype}"
)

    # 3. Configure XClone for this subset
    xconfig = xclone.XCloneConfig(dataset_name=safe_ct_name, module="Combine")

    xconfig.set_figure_params(xclone=True, fontsize=18)
    xconfig.outdir = current_outdir  # <--- Dynamic Output Directory

    xconfig.guide_chr_anno_key = "chr"
    xconfig.xclone_plot = True
    xconfig.plot_cell_anno_key = "sample"
    xconfig.merge_loss = True
    xconfig.merge_loh = True
    
    # Reference Settings
    # IMPORTANT: XClone will look for these reference samples WITHIN the current 'rdr_sub'.
    # If 'Embryo...38' has NO cells of this 'cell_type', XClone will fail to find a reference.
    xconfig.cell_anno_key = "dataset"
    xconfig.multi_refcelltype = True
    xconfig.ref_celltype = ["Embryo-Imp15265938", "Embryo-Imp15265940", "Embryo-Imp15265944"]
    
    xconfig.WGD_detection = False
    xconfig.BAF_denoise = False
    xconfig.RDR_denoise = False
    xconfig.display()

    # 4. Run Combine
    try:
        combine_Xdata = xclone.model.run_combine(
            rdr_sub,
            baf_sub,
            verbose=False,
            run_verbose=False,
            config_file=xconfig
        )

        # 5. Save
        # Ensure column conversion for saving
        if 'scrublet_prediction' in combine_Xdata.obs.columns:
            combine_Xdata.obs['scrublet_prediction'] = combine_Xdata.obs['scrublet_prediction'].astype(str)
        
        save_path = os.path.join(current_outdir, "Combined_merge_Xdata_KNN_HMM_post.h5ad")
        print(f"  > Saving to: {save_path}")
        combine_Xdata.write(save_path)

    except Exception as e:
        print(f"ERROR processing {cell_type}: {e}")
        print("Continuing to next cell type...")

    finally:
        # This always runs, success or error
        del rdr_sub, baf_sub, combine_Xdata, xconfig
        gc.collect()


print("\nBatch processing complete.")