library(dplyr)
library(tidyr)
library(readxl)
library(ggplot2)
library(purrr)
library(igraph)

### set working directory to root path "GLRP_BRCAness"
setwd("~/Downloads/BRCAness_project/Publication/GIT/brcaness_glrp_deciphering-main")  
UCSCTCGA_HRD_file <- "./data/TCGA.HRD_withSampleID.txt.gz"
Clinical_file <- "./data/Survival_SupplementalTable_S1_20171025_xena_sp"
UCSCMutation_M3C_file <- "./data/mc3.v0.2.8.PUBLIC.xena.gz"
TCGA_EXP_file <- "./data/tcga_gene_expected_count.gz"   ## expected count
GeneID_file<- "./data/probeMap_gencode.v23.annotation.gene.probemap"

# PanCan33_ssGSEA_ZScore_file  <- "./data/PanCan33_ssGSEA_1387GeneSets_NonZero_sample_level_Z.txt.gz"




## data from BRCAness landscape paper https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9738094/
BRCAness_TCGA_ClinVarPatho_file <- "./data/cells-11-03877-s001/Table S4.xlsx"
BRCAness_TCGA_CosmicPatho_file <- "./data/cells-11-03877-s001/Table S3.xlsx"
BRCA_genes<-c("CCNE1","PALB2","ATM", "ATR", "AURKA", "BAP1", "BARD1", "BLM", "BRCA1", "BRCA2", "BRIP1", "CDK12", "CHD4", "CHEK1", "CHEK2", "C11orf30", "ERCC1", "FANCA", "FANCC", "FANCD2", "FANCE", "FANCF",
              "FANCI", "KMT2A", "MRE11A", "MYC", "NBN", "PALB2", "PARP1", "PAXIP1", "PLK1", "PTEN", "RAD50", "RAD51", "RAD51B", "RAD51C", "RAD51D", "RAD52", "SAMHD1", "SHFM1", "TP53", "TP53BP1", "WEE1", "WRN")

reactome.FIs_file <- "./data/FIsInGene_061424_with_annotations.txt"  ## FI network
create_directories <- function(path) {
  # Split the path into components
  path_components <- unlist(strsplit(path, "/"))
  
  # Initialize a variable to keep track of the current path
  current_path <- "."
  
  # Iterate over each component and create the directory if it doesn't exist
  for (dir in path_components) {
    current_path <- file.path(current_path, dir)
    
    if (!dir.exists(current_path)) {
      dir.create(current_path)
      cat("Created directory:", current_path, "\n")
    } else {
      cat("Directory already exists:", current_path, "\n")
    }
  }
}
# path <- "./data/Data_EXP_LRPData/TCGA_ALL_expected_count"
# create_directories(path)

TCGA_dataInputPreprocess <- function(GeneID_file,TCGA_EXP_file,UCSCMutation_M3C_file,UCSCTCGA_HRD_file,Clinical_file){

  Clinical_DF <- as.data.frame(data.table::fread(Clinical_file,sep = "\t",check.names = F,header=T,stringsAsFactors = F))
  
  
  
  GeneID_DF <- data.table::fread(GeneID_file,sep = "\t",check.names = F,header=T,stringsAsFactors = F)
  GeneID_DF <- as.data.frame(GeneID_DF)
  GeneID_DF<- GeneID_DF[c("id","gene")]
  GeneID_DF<- GeneID_DF[order(GeneID_DF$id),]
  TCGA_PanCan_EXP <-  as.data.frame(data.table::fread(TCGA_EXP_file,sep = "\t",check.names = F,header=T,stringsAsFactors = F))
  TCGA_PanCan_EXP<- merge(TCGA_PanCan_EXP,GeneID_DF,by.x = "sample",by.y = "id",all.x = T)
  TCGA_PanCan_EXP[is.na(TCGA_PanCan_EXP$gene),]$gene <- TCGA_PanCan_EXP[is.na(TCGA_PanCan_EXP$gene),]$sample
  TCGA_PanCan_EXP[ ,c('id', 'sample')] <- list(NULL)
  
  ###### deal with duplicate genes
  duplicatedTCGA_PanCan_EXP <- TCGA_PanCan_EXP[duplicated(TCGA_PanCan_EXP$gene) | duplicated(TCGA_PanCan_EXP$gene,fromLast = T),]   ## deal with duplicate gene situation
  
  if(NROW(duplicatedTCGA_PanCan_EXP)!=0) {    ## deal with duplicate gene name situation.
    duplicatedTCGA_PanCan_EXP <- aggregate(. ~gene,duplicatedTCGA_PanCan_EXP,mean)
  } else {
    duplicatedTCGA_PanCan_EXP <- duplicatedTCGA_PanCan_EXP
  }
  
  uniqueTCGA_PanCan_EXP <- TCGA_PanCan_EXP[!(duplicated(TCGA_PanCan_EXP$gene) | duplicated(TCGA_PanCan_EXP$gene,fromLast = T)),]
  TCGA_PanCan_EXP <- rbind(uniqueTCGA_PanCan_EXP,duplicatedTCGA_PanCan_EXP)
  ####
  
  # PanCan33_ssGSEA_Zscore_DF <- as.data.frame(data.table::fread(PanCan33_ssGSEA_ZScore_file,sep = "\t",check.names = F,header=T,stringsAsFactors = F))
  # PanCan33_ssGSEA_Zscore_DF <- dplyr::rename(PanCan33_ssGSEA_Zscore_DF, Pathway = sample)
  
  UCSCMutation_M3C_DF <-as.data.frame(data.table::fread(UCSCMutation_M3C_file,sep = "\t",check.names = F,header=T,stringsAsFactors = F))
  UCSCTCGA_HRD <- as.data.frame(data.table::fread(UCSCTCGA_HRD_file,sep = "\t",check.names = F,header=T,stringsAsFactors = F))
  UCSCTCGA_HRD <- UCSCTCGA_HRD %>%
    pivot_longer(cols=c(-sampleID),names_to="sample")%>%
    pivot_wider(names_from=c(sampleID))
  
  # TCGA_phenotype_DF <- Phenotype_DF_Create(TCGA_phenotype_file,TCGA_abbr_file)
  
  KeepedSamples <-Reduce(intersect, list(colnames(TCGA_PanCan_EXP),UCSCMutation_M3C_DF$sample,UCSCTCGA_HRD$sample,Clinical_DF$sample))  ## samples hold exp, mut, HRD, phenotype info
  Clinical_DF <- Clinical_DF[Clinical_DF$sample %in% KeepedSamples,]
  TCGA_phenotype_DF <- Clinical_DF %>% dplyr::select(sample,`cancer type abbreviation`) %>%dplyr::rename(disease = `cancer type abbreviation`)
  UCSCTCGA_HRD <- UCSCTCGA_HRD[UCSCTCGA_HRD$sample %in% KeepedSamples,]
  TCGA_PanCan_EXP <- TCGA_PanCan_EXP[,c("gene",KeepedSamples)]
  # PanCan33_ssGSEA_Zscore_DF <- PanCan33_ssGSEA_Zscore_DF[,c("Pathway",intersect(KeepedSamples,colnames(PanCan33_ssGSEA_Zscore_DF)))]
  UCSCMutation_M3C_DF <- UCSCMutation_M3C_DF[UCSCMutation_M3C_DF$sample %in% KeepedSamples, ]
  
  TCGA_inputList = list(TCGA_PanCan_EXP = TCGA_PanCan_EXP,UCSCMutation_M3C_DF=UCSCMutation_M3C_DF,KeepedSamples=KeepedSamples,UCSCTCGA_HRD=UCSCTCGA_HRD,TCGA_phenotype_DF=TCGA_phenotype_DF,TCGA_EXP_file =TCGA_EXP_file,Clinical_DF =Clinical_DF)
  
  return(TCGA_inputList)
}
TCGA_inputList <- TCGA_dataInputPreprocess(GeneID_file,TCGA_EXP_file,UCSCMutation_M3C_file,UCSCTCGA_HRD_file,Clinical_file)
PPICreate <- function(reactome.FIs_file,IDtype) {
  if(missing(IDtype)) 
    IDtype = "UNIPROT"
  # ReactomePAPPI_file<-reactome.FIs_file
  GeneID_DF <- read.table(reactome.FIs_file,sep = "\t",check.names = F,header=T,comment.char = "")
  
  if("Gene1" %in% colnames(GeneID_DF) & "Gene2" %in% colnames(GeneID_DF) & "Score" %in% colnames(GeneID_DF)){
    GeneID_DF <- GeneID_DF[GeneID_DF$Score == 1,]
    reactome.ppi_data <- GeneID_DF[c("Gene1","Gene2")]
    return(reactome.ppi_data)
    
  }
  GeneID_DF <- GeneID_DF %>% dplyr::rename(`Interactor 1 uniprot id`=`# Interactor 1 uniprot id`)
  GeneID_DF_Uniport <- GeneID_DF
  # GeneID_DF_Uniport <- GeneID_DF[grepl("uniprotkb",GeneID_DF$`Interactor 1 uniprot id`,ignore.case = T) & grepl("uniprotkb",GeneID_DF$`Interactor 2 uniprot id`,ignore.case = T),]
  
  GeneID_DF_Uniport <- GeneID_DF_Uniport %>%
    mutate(across(c(`Interactor 1 uniprot id`,`Interactor 2 uniprot id`), gsub, pattern = "^.*?:", replacement = ""))
  GeneID_DF_Uniport <- GeneID_DF_Uniport[GeneID_DF_Uniport$`Interactor 1 uniprot id` != GeneID_DF_Uniport$`Interactor 2 uniprot id`, ]
  
  UniportIDconvertDF <-bitr(c(GeneID_DF_Uniport$`Interactor 1 uniprot id`,GeneID_DF_Uniport$`Interactor 2 uniprot id`), fromType="UNIPROT", toType="SYMBOL", OrgDb="org.Hs.eg.db")
  
  GeneID_DF_Uniport <- GeneID_DF_Uniport %>%
    inner_join(UniportIDconvertDF,by=c("Interactor 1 uniprot id" = "UNIPROT")) %>%
    inner_join(UniportIDconvertDF,by=c("Interactor 2 uniprot id" = "UNIPROT"))
  GeneID_DF_uniportID <-unique(GeneID_DF_Uniport[c("SYMBOL.x","SYMBOL.y")]) %>% dplyr::rename("Gene1" = "SYMBOL.x","Gene2" = "SYMBOL.y")
  
  
  GeneID_DF_Ensembl <- GeneID_DF[grepl("ENSEMBL",GeneID_DF$`Interactor 1 Ensembl gene id`,ignore.case = T) & grepl("ENSEMBL",GeneID_DF$`Interactor 2 Ensembl gene id`,ignore.case = T),]
  GeneID_DF_Ensembl <- GeneID_DF_Ensembl %>% separate_rows(`Interactor 1 Ensembl gene id`,sep = "\\|", convert = TRUE) %>% separate_rows(`Interactor 2 Ensembl gene id`,sep = "\\|", convert = TRUE)
  GeneID_DF_Ensembl <-unique(GeneID_DF_Ensembl[c("Interactor 1 Ensembl gene id","Interactor 2 Ensembl gene id")])
  GeneID_DF_Ensembl <- GeneID_DF_Ensembl %>%
    mutate(across(c(`Interactor 1 Ensembl gene id`,`Interactor 2 Ensembl gene id`), gsub, pattern = "^.*?:", replacement = ""))
  
  EnsemblIDconvertDF1 <-bitr(c(GeneID_DF_Ensembl$`Interactor 1 Ensembl gene id`,GeneID_DF_Ensembl$`Interactor 2 Ensembl gene id`), fromType=c("ENSEMBL"), toType=c("SYMBOL"), OrgDb="org.Hs.eg.db") %>%
    dplyr::rename(ENSEMBLID = ENSEMBL)
  EnsemblIDconvertDF2 <-bitr(c(GeneID_DF_Ensembl$`Interactor 1 Ensembl gene id`,GeneID_DF_Ensembl$`Interactor 2 Ensembl gene id`), fromType=c("ENSEMBLPROT"), toType="SYMBOL", OrgDb="org.Hs.eg.db") %>%
    dplyr::rename(ENSEMBLID = ENSEMBLPROT)
  EnsemblIDconvertDF3 <-bitr(c(GeneID_DF_Ensembl$`Interactor 1 Ensembl gene id`,GeneID_DF_Ensembl$`Interactor 2 Ensembl gene id`), fromType=c("ENSEMBLTRANS"), toType="SYMBOL", OrgDb="org.Hs.eg.db") %>%
    dplyr::rename(ENSEMBLID = ENSEMBLTRANS)
  EnsemblIDconvertDF <- do.call("rbind", list(EnsemblIDconvertDF1,EnsemblIDconvertDF2, EnsemblIDconvertDF3))
  GeneID_DF_Ensembl <- GeneID_DF_Ensembl[GeneID_DF_Ensembl$`Interactor 1 Ensembl gene id` != GeneID_DF_Ensembl$`Interactor 2 Ensembl gene id`,]
  
  
  GeneID_DF_Ensembl <- GeneID_DF_Ensembl %>%
    inner_join(EnsemblIDconvertDF,by=c("Interactor 1 Ensembl gene id" = "ENSEMBLID")) %>%
    inner_join(EnsemblIDconvertDF,by=c("Interactor 2 Ensembl gene id" = "ENSEMBLID"))
  GeneID_DF_EnsemblID <-unique(GeneID_DF_Ensembl[c("SYMBOL.x","SYMBOL.y")]) %>% dplyr::rename("Gene1" = "SYMBOL.x","Gene2" = "SYMBOL.y")
  
  GeneID_DFAll <-unique(rbind(GeneID_DF_uniportID,GeneID_DF_EnsemblID))
  GeneID_DFAll <- unique(GeneID_DFAll[GeneID_DFAll$Gene1 != GeneID_DFAll$Gene2,])
  
  if(grepl("UNIPROT",IDtype,ignore.case = T)){
    
    
    return(GeneID_DF_uniportID)
    
  } else if(grepl("ENSEMBL",IDtype,ignore.case = T) ){
    
    return(GeneID_DF_EnsemblID)
  } else {
    return(GeneID_DFAll)
  }
  # rbind()
}


TCGAPAN_PathoSamples_Create<- function(BRCAness_TCGA_ClinVarPatho_file,BRCAness_TCGA_CosmicPatho_file){
  BRCAness_TCGA_ClinVarPatho_DF <- read_excel(BRCAness_TCGA_ClinVarPatho_file,range = "A3:Q3166")
  BRCAness_TCGA_ClinVarPatho_DF <- BRCAness_TCGA_ClinVarPatho_DF %>% dplyr::select(-...9) %>% dplyr::rename(TCGA_Chr. = Chr....2,TCGA_Start = Start...3, ClinVar_Chr = Chr....10, ClinVar = Start...11) %>% 
    mutate(`TCGA sample barcode` = substr(Sample_Barcode, 1, 15)) 
  
  BRCAness_TCGA_CosmicPatho_DF <- read_excel(BRCAness_TCGA_CosmicPatho_file,range = "A3:Q3166")
  BRCAness_TCGA_CosmicPatho_DF <- BRCAness_TCGA_CosmicPatho_DF %>% dplyr::select(-...9) %>% mutate(`TCGA sample barcode` = substr(`Sample Barcode`, 1, 15)  ) 
  
  TCGA_pathoSamples <-union(BRCAness_TCGA_CosmicPatho_DF$`TCGA sample barcode`,BRCAness_TCGA_ClinVarPatho_DF$`TCGA sample barcode`)   ### TCGA pathogenetic mutation samples 
  TCGA_BRCA_pathoSamples <-union(BRCAness_TCGA_ClinVarPatho_DF[grepl("BRCA1|BRCA2",BRCAness_TCGA_ClinVarPatho_DF$Gene),]$`TCGA sample barcode`,
                                 BRCAness_TCGA_CosmicPatho_DF[grepl("BRCA1|BRCA2",BRCAness_TCGA_CosmicPatho_DF$Gene),]$`TCGA sample barcode`)     #### TCGA pathogenetic BRCA1/BRCA2 mutation samples 
  pathoSamples_list <- list(TCGA_pathoSamples = TCGA_pathoSamples,TCGA_BRCA_pathoSamples= TCGA_BRCA_pathoSamples)
  return(pathoSamples_list)
}

pathoSamples_list <- TCGAPAN_PathoSamples_Create(BRCAness_TCGA_ClinVarPatho_file,BRCAness_TCGA_CosmicPatho_file)


TCGA_BRCAness_Preprocess_UCSC <- function(TCGA_inputList,pathoSamples_list,reactome.FIs_file,Included_cancerType,Sample_Count_Threshold,BRCA_genes){
  if(missing(Included_cancerType)){
    Included_cancerType <- "ALL"
  }
  if(missing(Sample_Count_Threshold)){
    Sample_Count_Threshold <- 0
  }
  if(missing(BRCA_genes)) BRCA_genes <- NULL
  
  UCSCTCGA_HRD <- TCGA_inputList$UCSCTCGA_HRD
  TCGA_phenotype_DF <- TCGA_inputList$TCGA_phenotype_DF
  # test<- UCSCTCGA_HRD %>% inner_join(TCGA_phenotype_DF,by= join_by(sample)) %>% filter(disease == "BRCA") %>% summarise(mean(HRD),sd(HRD))
  
  
  BRCAness_samples_candidate1 <- union(UCSCTCGA_HRD[UCSCTCGA_HRD$HRD>=42,]$sample,pathoSamples_list$TCGA_BRCA_pathoSamples) ## >=42 or BRCA1/BRCA2 patho mut
  # # 
  BRCAness_samples_candidate2 <-intersect(UCSCTCGA_HRD[UCSCTCGA_HRD$HRD>=33,]$sample,pathoSamples_list$TCGA_pathoSamples) ## >=33 and BRCA gene list patho mut
  BRCAness_samples <- union(BRCAness_samples_candidate1,BRCAness_samples_candidate2)
  

  
  
  
  
  UCSCMutation_M3C_DF <- TCGA_inputList$UCSCMutation_M3C_DF
  
  # Candidate_pathoSamples <- UCSCMutation_M3C_DF[grepl("deleterious",UCSCMutation_M3C_DF$SIFT) & grepl("probably_damaging",UCSCMutation_M3C_DF$PolyPhen) & UCSCMutation_M3C_DF$gene %in% BRCA_genes ,]$sample
  # NonBRCAness_samples_cand <- setdiff(unique(TCGA_phenotype_DF$sample),union(pathoSamples_list$TCGA_pathoSamples,Candidate_pathoSamples)) #strict rule
  NonBRCAness_samples_candidate_pre <- setdiff(unique(TCGA_phenotype_DF$sample),pathoSamples_list$TCGA_pathoSamples) # remove the sample with no BRCAness associate gene mutations
  NonBRCAness_samples_candidate1<- union(UCSCTCGA_HRD[UCSCTCGA_HRD$HRD<33,]$sample,intersect(NonBRCAness_samples_candidate_pre,UCSCTCGA_HRD[UCSCTCGA_HRD$HRD<42,]$sample))

  NonBRCAness_samples <- NonBRCAness_samples_candidate1

  
  
  BRCAness_DF_cancerType <- TCGA_phenotype_DF %>% filter(sample %in% c(BRCAness_samples,NonBRCAness_samples) ) %>% mutate(BRCAness=ifelse(sample %in% BRCAness_samples, 1,0))
  BRCAness_DF_plot_pre <- BRCAness_DF_cancerType %>% dplyr::select(disease,BRCAness) %>%  mutate(BRCAness =ifelse(BRCAness == 1, "BRCAness","Non_BRCAness")) %>% add_count(disease,BRCAness) %>% distinct()
  
  BRCAness_DF_plot_expanded <- BRCAness_DF_plot_pre %>%
    complete(disease, BRCAness, fill = list(n = 0))
  
  resampled_cancerSubtype_plot_pre <- ggplot(BRCAness_DF_plot_expanded, aes(disease, n, fill = BRCAness)) +
    geom_bar(stat="identity", position = position_dodge(width=0.9)) +
    scale_fill_brewer(palette = "Set1") +
    scale_fill_brewer(palette = "Set1") + geom_text(position = position_dodge2(width= 0.9),aes(label = n),size= 5,hjust=0.5, vjust= -0.5) +
    labs(x = "Cancer Type", y = "Number of Cases", title = "Distribution of Cancer Subtypes by BRCAness Status Before Resampling") + # Rename x-axis title to "Disease" and y-axis title to "Count"
    theme(axis.text.x = element_text(face = "bold",size = 13), 
          axis.text.y = element_text(face = "bold",size = 12),
          axis.title.x = element_text(face = "bold", size = 14),
          axis.title.y = element_text(face = "bold", size = 14),
          plot.title = element_text(face = "bold", size = 16),  # Increase plot title size
          legend.title = element_text(size = 16),  # Increase legend title size
          legend.text = element_text(size = 14))   # Increase legend text size
   
  
  # print(resampled_cancerSubtype_plot_pre)
  # 
  # resampled_cancerSubtype_plot_pre <- ggplot(BRCAness_DF_plot_pre, aes(disease, n, fill = BRCAness)) +
  #   geom_bar(stat="identity", position = "dodge") +
  #   scale_fill_brewer(palette = "Set1") + geom_text(position = position_dodge2(width= 0.9),aes(label = n),size= 4,hjust=0.5, vjust= -0.5)
  #########################  Perform undersampling for each group
  undersample_group <- function(df) {
    if (n_distinct(df$BRCAness) < 2) {
      return(sample_n(df, 1)) # Randomly keep only one sample if the group has only one unique BRCAness value
    }
    
    min_class_count <- min(table(df$BRCAness))
    df_balanced <- df %>%
      group_by(BRCAness) %>%
      sample_n(min_class_count)
    return(df_balanced)
  }
  
  # Perform random undersampling on each disease group and combine the results
  BRCAness_DF <- BRCAness_DF_cancerType %>%
    group_by(disease) %>%
    nest() %>%
    mutate(balanced_data = map(data, undersample_group)) %>%
    dplyr::select(-data) %>%
    unnest(cols = c(balanced_data))
  
  BRCAness_DF <- as.data.frame(BRCAness_DF)
  # BRCAness_DF = NULL
  # curr_frame <<- sys.nframe()
  # for (c in unique(BRCAness_DF_cancerType$disease)) {
  #   tmp_df <<- BRCAness_DF_cancerType%>%filter(disease==c)
  #   if(sum(tmp_df$BRCAness)< 1){
  #     BRCAness_DF<-BRCAness_DF
  #   } else {
  #     tmp<-ovun.sample(BRCAness ~ ., data = tmp_df, method = "under",  seed = 5)$data
  #     BRCAness_DF<-rbind(BRCAness_DF, tmp)
  #     
  #   }
  # }
  
  
  BRCAness_DF_plot <- BRCAness_DF %>% dplyr::select(disease,BRCAness) %>%  mutate(BRCAness =ifelse(BRCAness == 1, "BRCAness","Non_BRCAness")) %>% add_count(disease,BRCAness) %>% distinct()
  resampled_cancerSubtype_plot <- ggplot(BRCAness_DF_plot, aes(disease, n, fill = BRCAness)) + 
    geom_bar(stat="identity", position = "dodge") + 
    scale_fill_brewer(palette = "Set1") + geom_text(position = position_dodge2(width= 0.9),aes(label = n),size= 4,hjust=0.5, vjust= -0.5,fontface = "bold") +
    labs(x = "Cancer Type", y = "Number of Cases", title = "Distribution of Cancer Subtypes by BRCAness Status After Resampling") +
    theme(axis.text.x = element_text(face = "bold",size = 13), 
          axis.text.y = element_text(face = "bold",size = 12),
          axis.title.x = element_text(face = "bold", size = 14),
          axis.title.y = element_text(face = "bold", size = 14),
          plot.title = element_text(face = "bold", size = 16),  # Increase plot title size
          legend.title = element_text(size = 16),  # Increase legend title size
          legend.text = element_text(size = 14))

  resampled_cancerSubtype_plot
  
  BRCAness_DF_plot_filter <- BRCAness_DF_plot %>%
    group_by(`disease`) %>%
    filter(!any(n < Sample_Count_Threshold))
  
  resampled_cancerSubtype_filter_plot <- ggplot(BRCAness_DF_plot_filter, aes(disease, n, fill = BRCAness)) + 
    geom_bar(stat="identity", position = "dodge") + 
    scale_fill_brewer(palette = "Set1") + geom_text(position = position_dodge2(width= 0.9),aes(label = n),size= 4,hjust=0.5, vjust= -0.5,fontface = "bold") +
    labs(x = "Cancer Type", y = "Number of Cases", title = "Distribution of Cancer Subtypes by BRCAness Status After Resample Filtering") +
    theme(axis.text.x = element_text(face = "bold",size = 14), 
          axis.text.y = element_text(face = "bold",size = 14),
          axis.title.x = element_text(face = "bold", size = 14),
          axis.title.y = element_text(face = "bold", size = 14),
          plot.title = element_text(face = "bold", size = 16),  # Increase plot title size
          legend.title = element_text(size = 16),  # Increase legend title size
          legend.text = element_text(size = 14))


  combined_plot <- resampled_cancerSubtype_plot_pre  / resampled_cancerSubtype_filter_plot
  combined_plot
  Cand_disease <- unique(BRCAness_DF_plot_filter$disease)
  # paste(Cand_disease,collapse ="|")
  # BRCAness_DF<- BRCAness_DF[grepl( paste(Cand_disease,collapse ="|"),BRCAness_DF$disease,ignore.case = T),]
  
  if(Sample_Count_Threshold!=0 & Included_cancerType == "ALL"){
    BRCAness_DF<- BRCAness_DF[grepl( paste(Cand_disease,collapse ="|"),BRCAness_DF$disease,ignore.case = T),]
    BRCAness_DF_combLabel <- BRCAness_DF %>%
      mutate(combined_label = paste(disease, BRCAness, sep="_")) %>% dplyr::select(sample,combined_label) %>% tidyr::spread(sample,combined_label)
    # BRCAness_DF_combLabel_DF <- BRCAness_DF_combLabel
    
    BRCAness_DF <- BRCAness_DF[c("sample","BRCAness")]
  } else if(Sample_Count_Threshold ==0 & Included_cancerType == "ALL"){
    BRCAness_DF <- BRCAness_DF[c("sample","BRCAness")]
  } else {
    BRCAness_DF<- BRCAness_DF[grepl(Included_cancerType,BRCAness_DF$disease,ignore.case = T),]
    BRCAness_DF <- BRCAness_DF[c("sample","BRCAness")]
    
  }
  BRCAness_Label_DF <- BRCAness_DF
  rownames(BRCAness_Label_DF) <- BRCAness_Label_DF$sample
  BRCAness_Label_DF$sample <- NULL
  BRCAness_Label_DF<-as.data.frame(t(BRCAness_Label_DF))
  
  
  
  TCGA_PanCan_EXP<-  TCGA_inputList$TCGA_PanCan_EXP[,c("gene",BRCAness_DF$sample)]
  # TCGA_PanCan_EXP <- TCGA_PanCan_EXP[!(TCGA_PanCan_EXP$gene %in% BRCACand_list$BRCA_Candidate_genes),]  ## remove the exp candidate genes
  
  
  reactome.ppi_data <- PPICreate(reactome.FIs_file,"ALL")
  networkGenes <- union(reactome.ppi_data[,1],reactome.ppi_data[,2])
  Connected_genes <- intersect(networkGenes,TCGA_PanCan_EXP$gene)
  reactome.ppi_data <- subset(reactome.ppi_data,reactome.ppi_data[,1] %in% Connected_genes & reactome.ppi_data[,2] %in% Connected_genes)
  
  graph_ppi = graph.data.frame(reactome.ppi_data, directed = FALSE, vertices = NULL) 
  ppi_components = components(graph_ppi) # The bigest component is visible in this variable
  
  ppi_comp_max = max(ppi_components$csize) # number of vertices in the biggest connected component
  main_component_genes = names(ppi_components$membership[ppi_components$membership==which.max(ppi_components$csize)]) # extracting the genes from the 1st cluster, if it is the biggest connected component.
  
  TCGA_PanCan_EXP = TCGA_PanCan_EXP[TCGA_PanCan_EXP$gene %in% main_component_genes, ]
  colnames(TCGA_PanCan_EXP)[1] <-  "probe"
  duplicatedTCGA_PanCan_EXP <- TCGA_PanCan_EXP[duplicated(TCGA_PanCan_EXP$probe) | duplicated(TCGA_PanCan_EXP$probe,fromLast = T),]
  
  if(NROW(duplicatedTCGA_PanCan_EXP)!=0) {    ## deal with duplicate gene name situation.
    duplicatedTCGA_PanCan_EXP <- aggregate(. ~probe,duplicatedTCGA_PanCan_EXP,mean)
  } else {
    duplicatedTCGA_PanCan_EXP <- duplicatedTCGA_PanCan_EXP
  }
  
  uniqueTCGA_PanCan_EXP <- TCGA_PanCan_EXP[!(duplicated(TCGA_PanCan_EXP$probe) | duplicated(TCGA_PanCan_EXP$probe,fromLast = T)),]
  TCGA_PanCan_EXP <- rbind(uniqueTCGA_PanCan_EXP,duplicatedTCGA_PanCan_EXP)
  
  TCGA_PanCan_EXP <- TCGA_PanCan_EXP %>% relocate(probe, .after = last_col())
  reactome.FIs_adjacencyMatrix<-get.adjacency(graph_ppi, sparse=FALSE)[TCGA_PanCan_EXP$probe,TCGA_PanCan_EXP$probe]
  diag(reactome.FIs_adjacencyMatrix) <- 0
  DF_list<-list("TCGA_PanCan_EXP" = TCGA_PanCan_EXP,"reactome_FIs_TB" = reactome.FIs_adjacencyMatrix, "BRCAness_Label" = BRCAness_Label_DF,resampled_cancerSubtype_plot = resampled_cancerSubtype_plot,resampled_cancerSubtype_plot_pre = resampled_cancerSubtype_plot_pre,
                BRCAness_DF_pre = BRCAness_DF_cancerType, resampled_cancerSubtype_filter_plot = resampled_cancerSubtype_filter_plot,combined_plot = combined_plot)
  
  base_dir <- dirname(dirname(TCGA_EXP_file))
  figures_dir <- file.path(base_dir, "figures")
  # Create the directory if it doesn't exist
  if (!dir.exists(figures_dir)) {
    dir.create(figures_dir, recursive = TRUE)
  }
  
  # Define the file path for the plot
  combined_plot_path <- file.path(figures_dir, "Distribution_BeforeAfter_Resampling.png")
  
  # Save the plot with high resolution
  ggsave(filename = combined_plot_path, plot = combined_plot, width = 7680/300, height = 4320/300, dpi = 300)
  
  BRCA_DFlist_Output<- function(DF_list,TCGA_EXP_file,Included_cancerType){
    exp_label <- paste0(tail(strsplit(tools::file_path_sans_ext(basename(TCGA_EXP_file)),split="_")[[1]],2),collapse = "_") ##get norm,tpm or expected count label
    # outputDir <- paste(base_dir,"/","Data_EXP_LRPData/TCGA_",Included_cancerType,"_",exp_label,sep = "")
    outputDir <- paste0(base_dir,"/","Data_EXP_LRPData/TCGA_BRCA")
    create_directories(outputDir)
    
    if(exists("BRCAness_DF_combLabel")) {
      BRCAness_DF_combLabel_path <- paste(outputDir,"/","CancerEntity_combLabel",".csv",sep = "")
      write.table(BRCAness_DF_combLabel,BRCAness_DF_combLabel_path, sep = ",",row.names = FALSE,col.names = TRUE)
    }
    
    TCGA_PanCan_EXP_path <- paste(outputDir,"/","TCGA_exp_EXP",".csv",sep = "")
    reactome_FIs_path <- paste(outputDir,"/","TCGA_reactome_FIs",".csv",sep = "")
    BRCAness_Label_path <- paste(outputDir,"/","TCGA_BRCAness_Label",".csv",sep = "")
    
    write.table(DF_list$TCGA_PanCan_EXP,TCGA_PanCan_EXP_path, sep = ",",row.names = FALSE,col.names = TRUE)
    write.table(DF_list$reactome_FIs_TB,reactome_FIs_path, sep = ",",row.names = FALSE,col.names = TRUE)
    write.table(DF_list$BRCAness_Label,BRCAness_Label_path , sep = ",",row.names = FALSE,col.names = TRUE)
    
  }
  
  BRCA_DFlist_Output(DF_list,TCGA_inputList$TCGA_EXP_file,Included_cancerType)
  return(DF_list)
}
Included_cancerType <- "ALL"
Sample_Count_Threshold <- 5
TCGA_PanCan_list <-TCGA_BRCAness_Preprocess_UCSC(TCGA_inputList,pathoSamples_list,reactome.FIs_file,Included_cancerType,Sample_Count_Threshold)
View(TCGA_PanCan_list$TCGA_PanCan_EXP)
View(TCGA_PanCan_list$reactome_FIs_TB)
View(TCGA_PanCan_list$BRCAness_Label)
dim(TCGA_PanCan_list$TCGA_PanCan_EXP)
dim(TCGA_PanCan_list$reactome_FIs_TB)
dim(TCGA_PanCan_list$BRCAness_Label)


## DGE analysis in BRCAness
library(EnhancedVolcano)
# setwd("~/Downloads/BRCAness_project/Publication/GIT/GLRP_BRCAness)  ### set working directory
BRCAness_Label_file <- "./Data_EXP_LRPData/TCGA_BRCA/TCGA_BRCAness_Label.csv"
TCGA_PanCan_EXP_file <- "./Data_EXP_LRPData/TCGA_BRCA/TCGA_exp_EXP.csv"
BRCAness_Label <- data.table::fread(BRCAness_Label_file,sep = ",",check.names = F,header=T,stringsAsFactors = F)
TCGA_PanCan_EXP <- data.table::fread(TCGA_PanCan_EXP_file,sep = ",",check.names = F,header=T,stringsAsFactors = F)
# BRCAness_Label <- TCGA_PanCan_list$BRCAness_Label
# TCGA_PanCan_EXP <- TCGA_inputList$TCGA_PanCan_EXP


BRCAness_DF <-as.data.frame(t(BRCAness_Label))
BRCAness_DF <- BRCAness_DF %>% dplyr::rename(BRCAness= 1) %>% tibble::rownames_to_column( "sample") %>% mutate(BRCAlabel = ifelse(BRCAness==1,"BRCAness","NonBRCAness")) %>% 
  arrange(BRCAlabel)

TCGA_PanCan_EXP <- TCGA_PanCan_EXP  %>% tibble::remove_rownames() %>% tibble::column_to_rownames(var = "probe") %>%  dplyr::select(BRCAness_DF$sample)
TCGA_PanCan_EXP  = 2^TCGA_PanCan_EXP -1
keep = rowSums(TCGA_PanCan_EXP>=10)>min(table(BRCAness_DF$BRCAlabel))  ## size of the smallest group

TCGA_PanCan_EXP = TCGA_PanCan_EXP[keep,]
group_list <- BRCAness_DF$BRCAlabel
# group_list <- ifelse(BRCAness_DF$BRCAness==1,"BRCAness","Non-BRCAness")
group_list = factor(group_list,levels = c("BRCAness","NonBRCAness"))
###1. DESeq2-----------
#
library(DESeq2)
logFC_cutoff <- 1
DESeq2_DEG_create <- function(TCGA_PanCan_EXP,group_list,logFC_cutoff,volcanoplot_path){
  
  expr = floor(TCGA_PanCan_EXP)
  # expr[1:4,1:4]
  colData <- data.frame(row.names =colnames(expr), 
                        condition=group_list)
  dds <- DESeqDataSetFromMatrix(
    countData = expr,
    colData = colData,
    design = ~ condition)
  dds <- DESeq(dds)
  
  res <- results(dds, contrast = c("condition",rev(levels(group_list))))
  resOrdered <- res[order(res$pvalue),] # ??????P?????????
  DEG <- as.data.frame(resOrdered)
  head(DEG)
  # ??????NA???
  DEG <- na.omit(DEG)
  
  if(missing(logFC_cutoff)){
    logFC_cutoff <- with(DEG,mean(abs(log2FoldChange)) + 2*sd(abs(log2FoldChange)) )
  }
  # logFC_cutoff <- 1.5
  DEG$change = as.factor(
    ifelse(DEG$pvalue < 0.05 & abs(DEG$log2FoldChange) > logFC_cutoff,
           ifelse(DEG$log2FoldChange > logFC_cutoff ,'UP','DOWN'),'NOT')
  )
  
  # head(DEG)
  # table(DEG$change)
  DESeq2_DEG <- DEG
  DE_genes <- DESeq2_DEG[DESeq2_DEG$change!="NOT",]
  DE_genes <- tibble::rownames_to_column(DE_genes, "id")
  
  
  # Volcano plot
  # Volcano plot with EnhancedVolcano
  volcano_plot <-   EnhancedVolcano(DESeq2_DEG,
                                    lab = rownames(DESeq2_DEG),
                                    x = "log2FoldChange",
                                    y = "pvalue",
                                    title = 'DEG Volcano plot',
                                    xlim = c(-2.5, 2.5),
                                    pCutoff = 0.05,
                                    FCcutoff = logFC_cutoff,
                                    pointSize = 3.0,
                                    labSize = 4.0,
                                    labFace = "bold"  
                                    )
  
  ggsave(volcanoplot_path, plot = volcano_plot, width = 15, height = 12, dpi=300)
  DESeq2_DEG_list <- list(DESeq2_DEG=DESeq2_DEG,DESeq2_DE_genes = DE_genes,VolcanoPlot = volcano_plot)
  return(DESeq2_DEG_list)
}
volcanoplot_path = "./figures/volcano_plot.png"
DESeq2_DEG_list <- DESeq2_DEG_create(TCGA_PanCan_EXP,group_list,logFC_cutoff,volcanoplot_path)
DESeq2_DE_genes <-DESeq2_DEG_list$DESeq2_DE_genes
DESeq2_DEG_list$VolcanoPlot


###2.edgeR---------
# expr = TCGA_PanCan_EXP
library(edgeR)

edgeR_DEG_create<- function(TCGA_PanCan_EXP,group_list,logFC_cutoff){
  expr = TCGA_PanCan_EXP
  dge <- DGEList(counts=expr,group=group_list)
  dge$samples$lib.size <- colSums(dge$counts)
  dge <- calcNormFactors(dge) 
  
  design <- model.matrix(~0+group_list)
  rownames(design)<-colnames(dge)
  colnames(design)<-levels(group_list)
  
  dge <- estimateDisp(dge,design)
  # dge <- estimateGLMCommonDisp(dge,design)
  # dge <- estimateGLMTrendedDisp(dge, design)
  # dge <- estimateGLMTagwiseDisp(dge, design)
  
  fit <- glmFit(dge, design)
  lrt <- glmLRT(fit, contrast=c(-1,1)) 
  
  DEG=topTags(lrt, n=nrow(exp))
  DEG=as.data.frame(DEG)
  # logFC_cutoff <- with(DEG,mean(abs(logFC)) + 2*sd(abs(logFC)) )
  
  if(missing(logFC_cutoff)){
    logFC_cutoff <- with(DEG,mean(abs(logFC)) + 2*sd(abs(logFC)) )
  }
  #logFC_cutoff <- 2
  DEG$change = as.factor(
    ifelse(DEG$PValue < 0.05 & abs(DEG$logFC) > logFC_cutoff,
           ifelse(DEG$logFC > logFC_cutoff ,'UP','DOWN'),'NOT')
  )
  head(DEG)
  table(DEG$change)
  edgeR_DEG <- DEG
  DE_genes <- edgeR_DEG[edgeR_DEG$change!="NOT",]
  DE_genes <- tibble::rownames_to_column(DE_genes, "id")
  
  # Volcano plot with EnhancedVolcano
  volcano_plot <- EnhancedVolcano(edgeR_DEG,
                                  lab = rownames(edgeR_DEG),
                                  x = "logFC",
                                  y = "PValue",
                                  title = 'Volcano plot',
                                  pCutoff = 0.05,
                                  FCcutoff = logFC_cutoff,
                                  pointSize = 3.0,
                                  labSize = 3.0
  )
  edgeR_DEG_list <- list(edgeR_DEG=edgeR_DEG,edgeR_DE_genes = DE_genes,VolcanoPlot = volcano_plot)
  return(edgeR_DEG_list)
}
edgeR_DEG_list <- edgeR_DEG_create(TCGA_PanCan_EXP,group_list,logFC_cutoff)
edgeR_DE_genes <- edgeR_DEG_list$edgeR_DE_genes
edgeR_DEG_list$VolcanoPlot

###3.limma----
library(limma)
limma_DEG_create<- function(TCGA_PanCan_EXP,group_list,logFC_cutoff){
  expr = TCGA_PanCan_EXP
  
  design <- model.matrix(~0+group_list)
  colnames(design)=levels(group_list)
  rownames(design)=colnames(exp)
  
  dge <- DGEList(counts=expr)
  dge <- calcNormFactors(dge)
  v <- voom(dge,design, normalize="quantile")
  fit <- lmFit(v, design)
  
  constrasts = paste(rev(levels(group_list)),collapse = "-")
  cont.matrix <- makeContrasts(contrasts=constrasts,levels = design) 
  fit2=contrasts.fit(fit,cont.matrix)
  fit2=eBayes(fit2)
  
  DEG = topTable(fit2, coef=constrasts, n=Inf)
  DEG = na.omit(DEG)
  if(missing(logFC_cutoff)){
    logFC_cutoff <- with(DEG,mean(abs(logFC)) + 2*sd(abs(logFC)) )
  }
  
  #logFC_cutoff <- 2
  DEG$change = as.factor(
    ifelse(DEG$P.Value < 0.05 & abs(DEG$logFC) > logFC_cutoff,
           ifelse(DEG$logFC > logFC_cutoff ,'UP','DOWN'),'NOT')
  )
  head(DEG)
  limma_DEG <- DEG
  DE_genes <- limma_DEG[limma_DEG$change!="NOT",]
  DE_genes <- tibble::rownames_to_column(DE_genes, "id")
  
  # Volcano plot with EnhancedVolcano
  volcano_plot <- EnhancedVolcano(limma_DEG,
                                  lab = rownames(limma_DEG),
                                  x = "logFC",
                                  y = "P.Value",
                                  title = 'Volcano plot',
                                  pCutoff = 0.05,
                                  FCcutoff = logFC_cutoff,
                                  pointSize = 3.0,
                                  labSize = 3.0
  )
  
  limma_DEG_list <- list(limma_DEG=limma_DEG,limma_DE_genes = DE_genes,VolcanoPlot = volcano_plot)
  return(limma_DEG_list)
}
limma_DEG_list <- limma_DEG_create(TCGA_PanCan_EXP,group_list,logFC_cutoff)
limma_DE_genes <- limma_DEG_list$limma_DE_genes
limma_DEG_list$VolcanoPlot



