# Data Directory README

Welcome to the data directory! This README file provides guidance on the datasets that need to be downloaded into this directory, along with their descriptions, download links, and additional information on the annotation mapping file for querying by gene.

## Required Datasets

Please download the following datasets into this directory:

### 1. UCSCTCGA_HRD_file
- **Description**: This file contains HRD (Homologous Recombination Deficiency) scores with sample IDs.
- **Download Link**: [TCGA.HRD_withSampleID.txt.gz](https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/TCGA.HRD_withSampleID.txt.gz)

### 2. Clinical_file
- **Description**: This is curated clinical data from the Pan-cancer Atlas paper titled "An Integrated TCGA Pan-Cancer Clinical Data Resource (TCGA-CDR) to drive high-quality survival outcome analytics". The paper highlights four types of carefully curated survival endpoints and recommends the use of the endpoints of OS (Overall Survival), PFI (Progression-Free Interval), DFI (Disease-Free Interval), and DSS (Disease-Specific Survival) for each TCGA cancer type.
- **Download Link**: [Survival_SupplementalTable_S1_20171025_xena_sp](https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/Survival_SupplementalTable_S1_20171025_xena_sp)

### 3. UCSCMutation_M3C_file
- **Description**: This file contains TCGA Unified Ensemble "MC3" mutation calls. The work is described in the paper "Scalable Open Science Approach for Mutation Calling of Tumor Exomes Using Multiple Genomic Pipelines" (Cell Syst. 2018 Mar 28;6(3):271-281.e7). 
  - **DOI**: [10.1016/j.cels.2018.03.002](https://doi.org/10.1016/j.cels.2018.03.002)
  - **PubMed PMID**: [29596782](https://pubmed.ncbi.nlm.nih.gov/29596782/)
- **Download Link**: [mc3.v0.2.8.PUBLIC.xena.gz](https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/mc3.v0.2.8.PUBLIC.xena.gz)

### 4. TCGA_EXP_file
- **Description**: This file contains TOIL RSEM expected_count data. The unit used is log2(expected_count+1).
- **Download Link**: [tcga_gene_expected_count.gz](https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/tcga_gene_expected_count.gz)

### 5. GeneID_file
- **Description**: This is the annotation mapping file (ProbeMap) which can mapping genes to probes, transcripts, or exons.
- **Download Link**: [probeMap_gencode.v23.annotation.gene.probemap](https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/probeMap%2Fgencode.v23.annotation.gene.probemap)


### 6. reactome.FIs_file
- **Description**: Functional interactions (FIs) derived from Reactome and other pathway and interaction databases.
- **Download Link**: [FIsInGene_061424_with_annotations.txt](https://reactome.org/download/tools/ReatomeFIs/FIsInGene_061424_with_annotations.txt.zip)
- **Note**: After downloading, unzip the file and place the contents in the data directory.


### 7. BRCAness_TCGA_ClinVarPatho_file and BRCAness_TCGA_CosmicPatho_file
- **Description**: Pathogenetic variants in TCGA matched in Cosmic and Clinivar database.
  - **DOI**: [10.3390/cells11233877](https://doi.org/10.3390/cells11233877)
  - **PubMed PMID**: [36497135](https://pubmed.ncbi.nlm.nih.gov/36497135/)
- **Download Link**: [cells-11-03877-s001](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9738094/bin/cells-11-03877-s001.zip)
- **Note**: After downloading, unzip the file to the cells-11-03877-s001 folder in the data directory.

### 8. Cancer Gene Census data
- **Description**: This is the Cancer Gene Census data.
- **Download Link**: [Cosmic_CancerGeneCensus_Tsv_v98_GRCh37.tar](https://cancer.sanger.ac.uk/cosmic/download/cosmic/v98/cancergenecensus)
- **Note**: Users must register and apply for access to Cosmic data for academic use. Once approved, download the data and place it into the data directory to enable GLRP top gene analysis.

## Notes

- **File Format**: Ensure that each file is downloaded in its specified format (e.g., `.gz`, `.zip`). Do not change the file extensions as it may affect subsequent processing.
- **Filename Matching**: Confirm that the filenames match exactly as specified above. This is crucial for seamless integration with scripts or workflows that may rely on these specific filenames.
- **Annotation Mapping File**: For the annotation mapping file (GeneID_file), make sure it is placed in this directory if you plan to query data by gene. This file is necessary for mapping probes, transcripts, or exons to their corresponding genes.
- **Unzipping Files**: For files that are in `.zip` format, make sure to unzip them and place the contents directly in the data directory. Specifically:
  - For the `reactome.FIs_file`, unzip the file and place the contents into the data directory.
  - For `BRCAness_TCGA_ClinVarPatho_file` and `BRCAness_TCGA_CosmicPatho_file`, unzip the file into the `cells-11-03877-s001` folder within the data directory.
- **Access Requirements**: For the Cancer Gene Census data, users must register and apply for access to Cosmic data for academic use. Once access is granted, download the file, extract its contents, and place them in the data directory.
- **Directory Structure**: Maintain the directory structure as outlined to ensure that scripts and workflows can locate the files correctly. This includes creating subdirectories if specified, like `cells-11-03877-s001`.
- **Support and Queries**: For any issues or questions regarding the datasets or the download process, refer to the respective publications or support contacts of the data repositories. Ensure you follow the guidelines provided by each data source for proper usage and citation.
