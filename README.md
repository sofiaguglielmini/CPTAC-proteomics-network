# Comparative Network Analysis of CPTAC Proteomics Data in Lung Squamous Cell Carcinoma

This repository contains a reproducible network-based analysis of CPTAC proteomics data in lung squamous cell carcinoma (LSCC), with the goal of comparing protein network structure between normal and tumor samples.

The analysis focuses on conditional dependence networks inferred via sparse Gaussian graphical models, with emphasis on network topology, hub stability and pathway enrichment.
  
## Data
  
- **Source**: CPTAC Pan-Cancer proteomics (LSCC cohort)
- **Level**: Gene-aggregated protein abundance
- **Filtering**: Genes in KEGG cancer pathways 
- **Preprocessing**: 
  - Missingness filtering and kNN imputation
  - Outlier removal
  - Scaling and centering

Raw CPTAC data are not included due to size and access constraints. The repository operates on processed data objects.
  
## Methods
  
- Sparse Gaussian graphical models (graphical lasso)
- Separate network estimation for normal and tumor samples
- Community detection via Louvain algorithm
- Centrality quantified by weighted degree
- Stability assessment via subsampling
- Comparison of node centrality and edges 
- Pathway analysis of non-small cell lung cancer pathway genes


## Repository structure

The full analysis and results are documented in `report.qmd`

To render it:
  
```bash
quarto render report.qmd
```
  
```
├── report.qmd          # Quarto report
├── src/
│   ├── data.py       # Data loading and preprocessing
│   ├── network.py    # Network estimation and metrics
│   └── visualize.py        # Plotting and visualization
├── run_analysis.py     # End-to-end analysis script
├── results/
│   ├── figures/        # Generated figures
│   └── tables/         # Summary tables
└── README.md
```
