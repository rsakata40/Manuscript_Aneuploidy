This repository contains analysis code associated with the manuscript
**"Distinct fates of aneuploid cells shape human blastoid development."**

## Repository structure

- `scRNAseq/` — single-cell RNA-sequencing processing, integration,
  aneuploidy inference and downstream analyses.
- `image_analysis/` — quantitative analysis of microscopy and imaging data.
- `environments/` — software environment files used for the analyses.

## scRNA-seq analysis

The `scRNAseq/` directory contains scripts and notebooks for:

1. Quality control and doublet detection
2. Per-sample scRNA-seq processing
3. Dataset integration and cell-type annotation
4. Aneuploidy inference using scploid
5. Celltype prediction using Early Embryo Prediction Tool (EEPT)
6. Phasing PacBio HiFi reads
7. Aneuploidy inference using XClone
8. Gene Expression analysis
9. Analysis of human embryo datasets

## Image analysis

The `image_analysis/` directory contains scripts used for quantitative
analysis of microscopy data, including developmental outcomes, cell number,
apoptosis, lineage markers and E-cadherin measurements.

## Software environments

Environment files required for selected analyses are available under
`environments/`.
