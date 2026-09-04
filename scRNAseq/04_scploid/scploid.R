# -*- coding: utf-8 -*-
library(scploid)
library(here)
library(biomaRt)
library(knitr)
library(BiocStyle)
library(Seurat)
library(tidyverse)
library(Matrix)
library(dplyr)
library(reshape2)
library(stringr)

# --- plotting theme (unchanged) ---
FONT.SIZE <- 9
LABEL.FONT.SIZE <- 9
w <- 2
h <- 2.5
LINE.W <- 0.3

settheme <- theme_minimal() + theme(
  panel.background  = element_blank(),
  panel.grid.major  = element_blank(),
  panel.grid.minor  = element_blank(),
  plot.background   = element_blank(),
  axis.ticks        = element_line(colour = "black", linewidth = LINE.W),
  axis.line         = element_line(linewidth = LINE.W, colour = "black"),
  axis.title        = element_text(size = FONT.SIZE),
  axis.text         = element_text(colour = "black", size = FONT.SIZE),
  axis.text.x       = element_text(colour = "black", angle = 0, size = LABEL.FONT.SIZE),
  legend.position   = "right",
  title             = element_text(size = FONT.SIZE)
)

# directories
figure_directory <- "/nfs/team292/rs40/projects/Aneuploid_screen_v2/processed_data/4_scploid"
data_directory   <- figure_directory

# metadata from Scanpy integration
meta_file   <- "/nfs/team292/rs40/projects/Aneuploid_screen_v2/processed_data/3_scanpy_integration/meta_integrated_sub.csv"
metadata.df <- read.csv(meta_file)

# samples
samples <- c(
  "Embryo-Imp15265938",
  "Embryo-Imp15265939",
  "Embryo-Imp15265940",
  "Embryo-Imp15265941",
  "Embryo-Imp15265942",
  "Embryo-Imp15265943",
  "Embryo-Imp15265944",
  "Embryo-Imp15265945",
  "Embryo-Imp15265946",
  "Embryo-Imp15265947"
)

# ------------------------------------------------------------------
# 1) Load 10x matrices, subset to Scanpy-filtered cells, make Seurat objects
# ------------------------------------------------------------------
seu_object_list <- list()

for (sample in samples) {
  # path to filtered_feature_bc_matrix
  mtx_dir <- file.path(
  "/lustre/scratch126/cellgen/vento/rs40/from_iRODs",
  paste0("cellranger900_count_49831_", sample, "_GRCh38-2024-A"),
  "filtered_feature_bc_matrix"
)
  setwd(mtx_dir)

  expression_matrix <- ReadMtx(
    mtx     = "matrix.mtx.gz",
    features = "features.tsv.gz",
    cells   = "barcodes.tsv.gz"
  )

  # match Scanpy ID format: Barcode_Sample
  colnames(expression_matrix) <- paste(colnames(expression_matrix), sample, sep = "_")

  # metadata IDs (must match this format)
  matching_cells <- colnames(expression_matrix)[colnames(expression_matrix) %in% metadata.df$cell.ID_]

  if (length(matching_cells) > 0) {
    expression_matrix <- expression_matrix[, matching_cells, drop = FALSE]
    seu_obj <- CreateSeuratObject(counts = expression_matrix)
    seu_object_list[[sample]] <- seu_obj
  } else {
    warning(paste("No matching cells found for sample:", sample))
  }
}

# Merge all Seurat objects
merged <- merge(
  seu_object_list[[1]],
  y = seu_object_list[seq(2, length(seu_object_list))]
)

# JoinLayers and extract raw counts matrix
obj <- JoinLayers(merged, assay = "RNA")
expression_matrix <- obj[["RNA"]]$counts  

# ------------------------------------------------------------------
# 2) Gene annotation via biomaRt (for scploid)
# ------------------------------------------------------------------
mart <- biomaRt::useMart(
  biomart = "ENSEMBL_MART_ENSEMBL",
  dataset = "hsapiens_gene_ensembl",
  host    = "http://www.ensembl.org"
)

gene_table <- getBM(
  attributes = c("external_gene_name", "chromosome_name"),
  mart       = mart,
  values     = as.character(rownames(expression_matrix)),
  filters    = "external_gene_name"
)

# autosomes only, drop duplicate gene names
gene_table <- gene_table[gene_table$chromosome_name %in% 1:22, ]
gene_table <- gene_table[!duplicated(gene_table$external_gene_name), ]

# keep only genes present in both matrix and gene_table
common_genes       <- intersect(rownames(expression_matrix), gene_table$external_gene_name)
gene_table         <- gene_table[gene_table$external_gene_name %in% common_genes, ]
expression_matrix  <- expression_matrix[gene_table$external_gene_name, ]

# ------------------------------------------------------------------
# 3) Order metadata to match expression_matrix columns
# ------------------------------------------------------------------
metadata_ordered <- metadata.df[match(colnames(expression_matrix), metadata.df$cell.ID_), ]

# optional sanity check
if (any(is.na(metadata_ordered$cell.ID_))) {
  stop("Some cells in expression_matrix do not match metadata.df$cell.ID_")
}

# ------------------------------------------------------------------
# 4) Run scploid
# ------------------------------------------------------------------
ploidytest <- makeAneu(
  counts     = as.matrix(expression_matrix),
  genes      = gene_table$external_gene_name,
  chrs       = gene_table$chromosome_name,
  cellNames  = colnames(expression_matrix),
  cellGroups = metadata_ordered$celltype_coarse   # per-cell group (e.g. EPI/TE/HYPO)
)

ploidytest <- setParam(ploidytest, param_name = "p.thresh", param_value = 0.01)
ploidytest <- doAneu(ploidytest)

ploidydf <- ploidytest@scores
ploidydf$ploidy <- "normal"
ploidydf$ploidy[ ploidydf$monosomy & ploidydf$p.adj < 1e-2 & ploidydf$score <= 0.8 ] <- "monosomy"
ploidydf$ploidy[ !ploidydf$monosomy & ploidydf$p.adj < 1e-2 & ploidydf$score >= 1.2 ] <- "trisomy"
ploidydf$chr <- as.factor(ploidydf$chr)

write.csv(
  ploidydf,
  "/nfs/team292/rs40/projects/Aneuploid_screen_v2/processed_data/4_scploid/ploidydf.csv",
  row.names = FALSE
)

message("scploid run finished successfully.")