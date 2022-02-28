#### Install dependencies and package (older version) ####
# install dependencies package
#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")
#BiocManager::install(version = "3.12")

# # install older version
# BiocManager::install(version="3.11")
# BiocManager::install("DESeq")
# 
# BiocManager::install(c("biomaRt",
#                        "circlize",
#                        "ComplexHeatmap",
#                        "corrplot",
#                        "DESeq2",
#                        "dplyr",
#                        "DT",
#                        "edgeR",
#                        "ggplot2",
#                        "limma",
#                        "lsmeans",
#                        "reshape2",
#                        "spatstat",
#                        "survival",
#                        "plyr"))
# 
# # install package
# install.packages("R_package/IMvigor210CoreBiologies_1.0.0.tar.gz", 
#                  repos=NULL)



#### Load libraries + data ####

library("IMvigor210CoreBiologies")
library(tidyverse)

data(cds)
counts <- counts(cds) # gene expression data
gene_an <- fData(cds) # annotation of genes, symbol, entrez ID
clinical <- pData(cds) # clinical data of patients
data(fmone) # mutation data
mut_summary <- pData(fmone) # summary of mutation data


#### Count Data ####
# count data with samples on rows and samples as columns
counts_t <- t(counts)
counts_tib <- as_tibble(counts_t, rownames = "Sample")

# check if there are columns with only zeros and remove those
col_sums <- colSums(counts_tib %>% select(-Sample))
select_col <- which(col_sums != 0)
counts_wz <- counts_tib %>% select(-Sample) %>% select(all_of(select_col))
counts_wz["Sample"] <- counts_tib[1]


#### save separate and combined data ####
# Create a tibble of clinical features + drop samples of which binary response is NA or IC level is NA
clin_tib <- as_tibble(clinical, rownames = "Sample")
clin_tib <- clin_tib %>% filter(!is.na(binaryResponse)) %>% filter(!is.na(`IC Level`))

# convert categorical variables to integers
library(plyr)
clin_tib$binaryResponse <- ifelse(clin_tib$binaryResponse == "CR/PR", 1, 0)
clin_tib$Sex <- ifelse(clin_tib$Sex == "F", 1, 0)
clin_tib$`Intravesical BCG administered` <- ifelse(clin_tib$`Intravesical BCG administered` == "Y", 1, 0)
clin_tib$`IC Level` <- mapvalues(clin_tib$`IC Level`, from = c("IC0", "IC1", "IC2+"), to = c(0, 1, 2))
clin_tib$`TC Level` <- mapvalues(clin_tib$`TC Level`, from = c("TC0", "TC1", "TC2+"), to = c(0, 1, 2))


# Mutation burden
mut_overview <- as_tibble(mut_summary, rownames= "Sample") 
mut_overview$binaryResponse <- ifelse(mut_overview$binaryResponse == "CR/PR", 1, 0)
mut_overview <- mut_overview %>% filter(!is.na(binaryResponse)) %>% filter(!is.na(`IC Level`))
mut_burden <- mut_overview %>% select(c(Sample, `FMOne mutation burden per MB`, `ANONPT_ID`, binaryResponse))

# combine mutation data, count data and clinical features
clin_features <- clin_tib %>% select(c(Sample, binaryResponse, Sex, `Intravesical BCG administered`,
                                       `Baseline ECOG Score`, `ANONPT_ID`, `IC Level`, `TC Level`))
clin_features$ANONPT_ID <- as.integer(clin_features$ANONPT_ID)
clin_count <- left_join(clin_features, counts_wz, by="Sample")

# combine clinical features and immune phenotypes
clin_IP <- clin_tib %>% select(c(Sample, binaryResponse, Sex, `Intravesical BCG administered`,
                                       `Baseline ECOG Score`, `ANONPT_ID`, `IC Level`, `TC Level`, `Immune phenotype`)) %>% 
  filter(!is.na(`Immune phenotype`))
clin_IP$ANONPT_ID <- as.integer(clin_IP$ANONPT_ID)

# save expression data
bin_resp_sample <- clin_features %>% select(binaryResponse, Sample)
count_features <- inner_join(counts_wz, bin_resp_sample, by = "Sample")
# again remove columns with only zeros (after drop of certain patients (where no response was available) more columns only contain zeros)
col_sums <- colSums(count_features %>% select(-c(Sample, binaryResponse)))
select_col <- which(col_sums != 0)
counts_nz <- count_features %>% select(-c(Sample, binaryResponse)) %>% select(all_of(select_col))
counts_nz$Sample <- count_features$Sample
counts_nz$binaryResponse <- count_features$binaryResponse

## Do vst transformation of count data with DESeq2 package
#### Import libraries
library("DESeq2")
library("tidyverse")
library("EnhancedVolcano")
library("ggplot2")
library("gplots")
library("org.Hs.eg.db")
library("clusterProfiler")

# select the condition to split the two groups
response <- counts_nz['binaryResponse']
response$binaryResponse <- factor(response$binaryResponse, ordered=FALSE)

# transform count data to have genes on first column
counts_nt <- tibble(counts_nz)
condition_check <- counts_nt %>% select(c(Sample, binaryResponse))

counts <- counts_nt %>% select(-binaryResponse) %>% pivot_longer(!Sample, names_to = "genes", values_to = "counts")
counts <- counts %>% pivot_wider(names_from = Sample, values_from = counts)

as_matrix <- function(x){
  if(!tibble::is_tibble(x) ) stop("x must be a tibble")
  y <- as.matrix.data.frame(x[,-1])
  rownames(y) <- x[[1]]
  y
}

counts <- as_matrix(counts)
genes <- matrix(row.names(counts))
rownames(genes) <- genes

# create right input for DESeq2 - count data and data to group samples on
dds <- DESeqDataSetFromMatrix(countData = counts,
                              colData = response,
                              design = ~ binaryResponse)

# keep only rows with gene counts higher than 10
keep <- rowSums(counts(dds)) >= 10 
dds <- dds[keep, ]

# varianceStabilizingTransformation
varst <- vst(dds)
## retrieve variance transformed data from varst
vst_mat <- assay(varst)
#colnames(vst_mat) <- count_features$Sample
vst_mat <- t(vst_mat)

count_vst <- as_tibble(vst_mat, rownames = "Sample")
count_vst$binaryResponse <- as.factor(count_features$binaryResponse)

### Save everything with ANONPT_ID
# tumor mutation burden
TMB_ID <- mut_burden %>% select(`FMOne mutation burden per MB`, binaryResponse, ANONPT_ID)
TMB_ID$`FMOne mutation burden per MB` <- as.numeric(TMB_ID$`FMOne mutation burden per MB`)
# save TMB with clinical features (from this file TMB only can be taken)
TMB_clin_ID <- inner_join(TMB_ID %>% select(-binaryResponse), 
                       clin_features %>% select(-Sample), by="ANONPT_ID")
write.table(TMB_clin_ID, file="TMB_clin_ID.csv", sep=",")

# save TMB with clin features and immune phenotypes
TMB_clin_IP_ID <- inner_join(TMB_ID %>% select(-binaryResponse), 
                          clin_IP %>% select(-Sample), by="ANONPT_ID")
write.table(TMB_clin_IP_ID, file="TMB_clin_IP_ID.csv", sep=",")

## create datafile with clinical features, TMB and count features
clin_count <- left_join(clin_features, count_vst %>% select(-binaryResponse), by="Sample")
clin_count$ANONPT_ID <- as.numeric(clin_count$ANONPT_ID)
count_TMB_clin_ID <- inner_join(TMB_ID, clin_count %>% 
                                  select(-c(Sample, binaryResponse)), by="ANONPT_ID")
write.table(count_TMB_clin_ID, file="count_TMB_clin_ID.csv", sep=",")

## datafile with only count data + ANONPT_ID
count_vst_ID <- left_join(clin_features %>% select(c(Sample, ANONPT_ID)), 
                          count_vst, by="Sample")
write.table(count_vst_ID %>% select(-Sample), "count_vst_ID.csv", sep=",")

## create datafile with clinical features, immune phenotypes, TMB and count features
count_TMB_clin_IP_ID <- inner_join(TMB_clin_IP_ID, count_vst_ID %>% 
                                     select(-c(Sample, binaryResponse)), by="ANONPT_ID")
write.table(count_TMB_clin_IP_ID, file="count_TMB_clin_IP_ID.csv", sep=",")

## datafile with count data + ANONPT_ID + immune phenotypes
count_IP_ID <- left_join(clin_IP %>% select(c(Sample, ANONPT_ID, `Immune phenotype`)), 
                          count_vst, by="Sample")
write.table(count_IP_ID %>% select(-Sample), "count_vst_ID.csv", sep=",")


