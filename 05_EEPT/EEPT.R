#load packages
rm(list=ls())
suppressMessages({

  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(data.table)
  library(tibble)
  
  library(miloR)
  library(SingleCellExperiment)

  library(scater)
  library(scran)
  library(igraph)

  library(batchelor)
  library(uwot)

})

# CHANGE 1: Removed '~' to use absolute path. Safer for cluster jobs.
source("/nfs/team292/rs40/projects/Aneuploid_screen/notebooks/EEPT/HumanEarlyEmbryoRefProjection/HuEm_stable_reference_projection_tool/main.function.R")

# define const for visualization
FONT.SIZE <- 9
LABEL.FONT.SIZE <- 9
w <- 2
h <- 2.5
LINE.W <- 0.3 

settheme <- theme_minimal() + theme(
    panel.background = element_blank(),
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    plot.background = element_blank(),
    axis.ticks = element_line(colour = "black", linewidth = LINE.W),
    axis.line = element_line(linewidth = LINE.W, colour = "black"),
    axis.title = element_text(size = FONT.SIZE),
    axis.text = element_text(colour = "black", size = FONT.SIZE),
    axis.text.x = element_text(colour = "black", angle = 0,size = LABEL.FONT.SIZE),
    legend.position="right",
    title = element_text(size = FONT.SIZE) )

#settings for pipeline
para.list <- list()
para.list$cellGroup <- FALSE ## whether providing group information for cells
para.list$runMiloR <- TRUE  ## whether run miloR aggregation 
para.list$cor.cutoff <- 0.5 ## correlation cutoff

#Create Milo of query cells
demo_exp_file_path <- '/nfs/team292/rs40/projects/Aneuploid_screen_v2/processed_data/3_scanpy_integration/GEM_forEEPT.csv'
#demo_cell_group_file_path <- "demo/n1000.cell.group.meta.tsv" 

#loading example gene expression matrix
# CHANGE 2: Added check.names=FALSE to preserve exact barcode spelling
temp.counts <-  fread(demo_exp_file_path, header = T, sep=",", check.names = FALSE) %>% 
  tibble::column_to_rownames("Gene") 

#creating or loading cell meta
if (para.list$cellGroup) {
  temp.counts.meta <- read.delim(demo_cell_group_file_path, head=F, stringsAsFactors = F, sep="\t") %>% 
    tbl_df() %>% 
    setNames(c("cell","group")) %>% 
    mutate(pj="query", EML="query")
    
    # Ensure rownames match here too just in case
    rownames(temp.counts.meta) <- temp.counts.meta$cell
    
} else {
  # CHANGE 3: The Critical Fix for preserving ID names
  cell_ids <- colnames(temp.counts)
  
  temp.counts.meta <- data.frame(cell = cell_ids) %>% 
    tbl_df() %>% 
    mutate(cell = as.vector(cell)) %>% 
    mutate(EML="query", group="None", pj="query")
    
  # THIS LINE IS ESSENTIAL to keep IDs from becoming 1, 2, 3...
  rownames(temp.counts.meta) <- cell_ids
}

# Ensure alignment just to be safe
temp.counts.meta <- temp.counts.meta[colnames(temp.counts), ]

#creating milo object
if (para.list$runMiloR) {
  milo_out <- FunMiloCal(temp.counts, temp.counts.meta, temp.cal=TRUE)
} else {
  milo_out <- FunMiloCal(temp.counts, temp.counts.meta, temp.cal=FALSE)
}

#generating comparable size factor for query datasets
sf_out <- FunCalSF(milo_out)

#calculating correlation with reference cells
sf_out$query.sce.cor.out <- FunCalCor(sf_out) 
sf_out$query.sce.cor.out.mean <- sf_out$query.sce.cor.out %>% 
  gather(ref_cell,cor,-query_cell) %>% 
  group_by(query_cell) %>% 
  top_n(20,cor) %>% 
  summarise(cor_top_mean=mean(cor))

#generate downsampling samples
sf_out$NWIN <- FunNWIN(sf_out)

#MNN calculation for each datasets
mnn.pairs.list <- list()
for (ref_name in c("SPH2016","D3post","Meistermann_2021","CS7","nBGuo","Yan2013")) {
  mnn.pairs.list[[ref_name]] <- FunCalMNN_each(sf_out, ref_name)
}

#generating the 2D projection and 20D embeddings in latent space
predict_out <-  FunProjCal(mnn.pairs.list,
                           query.sce.ob=sf_out$query.sce.ob,
                           query.sce.cor.out=sf_out$query.sce.cor.out,
                           temp.max=sf_out$NWIN$temp.max,
                           D2_umap_model=ref.umap,
                           Dmulti_umap_model=ref_umap_nDim_model,
                           cor.cutoff=para.list$cor.cutoff)

#including raw meta information
predict_out$HS <- milo_out$HS
predict_out$raw.meta <- milo_out$raw.meta
predict_out$query.sce.cor.out.mean <- sf_out$query.sce.cor.out.mean

#cell identities prediction
predict_out <- FunPredAnno(predict_out, cor.cutoff=para.list$cor.cutoff)

#check the prediction results
print("Prediction Summary:")
print(predict_out$full.anno %>% group_by(pred_EML) %>% summarise(nCell=n_distinct(query_cell)) %>% arrange(desc(nCell)))

# save
saveRDS(predict_out, file = "/nfs/team292/rs40/projects/Aneuploid_screen_v2/processed_data/6_EEPT/predicted_EEPT.rds")