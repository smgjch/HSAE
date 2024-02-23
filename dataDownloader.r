if (!requireNamespace("BiocManager", quietly=TRUE))
  install.packages("BiocManager")
BiocManager::install("TCGAbiolinks")

library(TCGAbiolinks)
library(dplyr)
library(DT)


# You can define a list of samples to query and download providing relative TCGA barcodes.
listSamples <- c(
    "TCGA-E9-A1NG-11A-52R-A14M-07","TCGA-BH-A1FC-11A-32R-A13Q-07",
    "TCGA-A7-A13G-11A-51R-A13Q-07","TCGA-BH-A0DK-11A-13R-A089-07",
    "TCGA-E9-A1RH-11A-34R-A169-07","TCGA-BH-A0AU-01A-11R-A12P-07",
    "TCGA-C8-A1HJ-01A-11R-A13Q-07","TCGA-A7-A13D-01A-13R-A12P-07",
    "TCGA-A2-A0CV-01A-31R-A115-07","TCGA-AQ-A0Y5-01A-11R-A14M-07"
)

# Query platform Illumina HiSeq with a list of barcode 
query <- GDCquery(
    project = "TCGA-BRCA", 
    data.category = "Transcriptome Profiling",
    data.type = "Gene Expression Quantification",
    barcode = listSamples
)

# Download a list of barcodes with platform IlluminaHiSeq_RNASeqV2
TCGAbiolinks.GDCdownload(query)

# Prepare expression matrix with geneID in the rows and samples (barcode) in the columns
# rsem.genes.results as values
BRCA.Rnaseq.SE <- TCGAbiolinks.GDCprepare(query)

BRCAMatrix <- assay(BRCA.Rnaseq.SE,"unstranded") 
# For gene expression if you need to see a boxplot correlation and AAIC plot to define outliers you can run
BRCA.RNAseq_CorOutliers <- TCGAbiolinks.TCGAanalyze_Preprocessing(BRCA.Rnaseq.SE)