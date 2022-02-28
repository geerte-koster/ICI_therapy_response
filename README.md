# ICI_therapy_response
Repository containing code and files used to predict ICI therapy response of urothelial cancer patients

# Content
1) [train_and_validate_RF.py](100_repetitions_clf.py)
  <br/>Python file to train random forest classifier and save valdiation performance

2) [figures.py](figures.py)
<br/>Python file to reproduce all figures from study.

3) [data_preprocessing.R](data_preprocessing.R)
<br/>R file to preprocess data downloaded via IMvigor210CoreBiologies package of Mariathasan et al., 2018

4) [input_files](input_files)
<br/>Csv files containing various (combinations of) input features of which performance is described in Supplementary Table 2:
- TMB
- clincial features (including immune phenotypes)
- gene expression values (reduced using gene signatures)

5) [gene_signatures](gene_signatures)
<br/> Files with gene expression values of genes from signatures derived from Mariathasan et al., 2018 (CTL_signature*.csv), Jiang et al., 2018 (TIDE*.csv) and Litchfield et al., 2021 (litt_genes.csv), all signatures are listed in Supplementary Table 4
