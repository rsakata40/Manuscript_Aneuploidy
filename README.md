This repository contains analysis code associated with the manuscript
**"Distinct fates of aneuploid cells shape human blastoid development."**

## Repository structure

- directories with prefix `sc_` — includes codes for single-cell RNA-sequencing processing, and aneuploidy inference.
- directories with prefix `img_` — includes codes for quantitative analysis of microscopy and imaging data.
- `environments` — software environment files used for the analyses.

## scRNA-seq analysis

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

1. Plots of BF(brightfield) counts of developed and failed blastoids
2. Analysis of EGFP and mcherry cells in blastoids
3. Analysis of apoptosis in blastoids
4. Analysis of marker gene expression in blastoids
5. Plots of BF counts of developed and failed blastoids, with different number of reversine treated cells
6. Plots of BF counts of developed and failed blastoids, with different proportion of reversine treated cells
7. Plots of BF counts of developed and failed blastoids, with apoptosis inhibition (using ZVAD-FMK)
8. Analysis of cell counts and marker genes with apoptosis inhibition
9. Analysis of intensity of E-cadherin levels along cell boundary 
10. Analysis of marker genes of blastoids formed with CDH1 targeting siRNA
11. Plots of BF counts of developed and failed blastoids, with E-cadherin over expression
12. Analysis of marker genes of blastoids formed with E-cadherin over expression
13. Anaysis of marker genes in aneuploid human embryos
14. Analysis of marker genes in developed and poor quality human embryos

## Software environments

Environment files required for selected analyses are available under
`environments/`.
