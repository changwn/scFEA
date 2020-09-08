## scFEA: A graph neural network model to estimate cell-wise metabolic using single cell RNA-seq data scFEA

## Abstract

The metabolic heterogeneity, and metabolic interplay between cells and their microenvironment have been known as significant contributors to disease treatment resistance. Our understanding of the intra-tissue metabolic heterogeneity and cooperation phenomena among cell populations is unfortunately quite limited, without a mature single cell metabolomics technology. To mitigate this knowledge gap, we developed a novel computational method, namely **scFEA** (**s**ingle **c**ell **F**lux **E**stimation **A**nalysis), to infer single cell fluxome from single cell RNA-sequencing (scRNA-seq) data.  scFEA is empowered by a comprehensively reorganized human metabolic map as focused metabolic modules, a novel probabilistic model to leverage the flux balance constraints on scRNA-seq data, and a novel graph neural network based optimization solver. The intricate information cascade from transcriptome to metabolome was captured using multi-layer neural networks to fully capitulate the non-linear dependency between enzymatic gene expressions and reaction rates. We experimentally validated scFEA by generating an scRNA-seq dataset with matched metabolomics data on cells of perturbed oxygen and genetic conditions. Application of scFEA on this dataset demonstrated the consistency between predicted flux and metabolic imbalance with the observed variation of metabolites in the matched metabolomics data. We also applied scFEA on publicly available single cell melanoma and head and neck cancer datasets, and discovered different metabolic landscapes between cancer and stromal cells. The cell-wise fluxome predicted by scFEA empowers a series of downstream analysis including identification of metabolic modules or cell groups that share common metabolic variations, sensitivity evaluation of enzymes with regards to their impact on the whole metabolic flux, and inference of cell-tissue and cell-cell metabolic communications.

## The computational framework of scFEA

<p align="center">
  <img width="60%" src="https://github.com/changwn/scFEA/blob/master/doc/Figure%201.png">
</p>

## Supplementary Figures and Tables

[donwload supplementary files](https://github.com/changwn/scFEA/tree/master/supplementary%20data)

Supplementary Tables:

- Table S1. Information of reorgnized human metabolic map.

- Table S2. Metabolomics data and clusters of metabolic modules derived in the Pa03c cell line data.

- Table S3. Differentially expressed genes (DEG) and Pathway Enrichment (PE) results of the Pa03c cell line data.

- Table S4. Predicted cell type specific fluxome and metabolic imbalance in the melanoma and head and neck cancer data.

Supplementary Figures:

- Figure S1. The impact of each gene to the metabolic module 1-14 (glycolysis and TCA cycle modules) in the Pa03c cell line data.


## Questions & Problems

If you have any questions or problems, please feel free to open a new issue [here](https://github.com/changwn/scFEA/issues). We will fix the new issue ASAP.  You can also email the maintainers and authors below.

- [Wennan Chang](https://changwn.github.io/)
(wnchang@iu.edu)

PhD candidate at [Biomedical Data Research Lab (BDRL)](https://zcslab.github.io/) , Indiana University School of Medicine
