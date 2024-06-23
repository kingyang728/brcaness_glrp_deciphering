library(tidyr)
library(dplyr)
library("reactome.db")
library(org.Hs.eg.db)
library("KEGGREST")
library(clusterProfiler)
library(ReactomePA)

setwd("~/Downloads/BRCAness_project/Publication/GIT/brcaness_glrp_deciphering-main")

LRP_result_path <- "./results"
TCGA_GCNNInput_path <- "./Data_EXP_LRPData/TCGA_BRCA"
PathwayResult_Output_dir <- "./Heatmap_Network_result"

TCGA_PanCan_EXP_path <- paste0(TCGA_GCNNInput_path,"/TCGA_exp_EXP.csv")
BRCAness_Label_path <- paste0(TCGA_GCNNInput_path,"/TCGA_BRCAness_Label.csv") 
reactome_FIs_TB_path <- paste0(TCGA_GCNNInput_path,"/TCGA_reactome_FIs.csv")  

BRCAness_Label <- data.table::fread(BRCAness_Label_path,sep = ",",check.names = F,header=T,stringsAsFactors = F)
TCGA_PanCan_EXP <- data.table::fread(TCGA_PanCan_EXP_path,sep = ",",check.names = F,header=T,stringsAsFactors = F)
reactome_FIs_TB <- as.matrix(data.table::fread(reactome_FIs_TB_path,sep = ",",check.names = F, header=T,stringsAsFactors = F))
BRCA_genes<-c("CCNE1","PALB2","ATM", "ATR", "AURKA", "BAP1", "BARD1", "BLM", "BRCA1", "BRCA2", "BRIP1", "CDK12", "CHD4", "CHEK1", "CHEK2", "C11orf30", "ERCC1", "FANCA", "FANCC", "FANCD2", "FANCE", "FANCF",
              "FANCI", "KMT2A", "MRE11A", "MYC", "NBN", "PALB2", "PARP1", "PAXIP1", "PLK1", "PTEN", "RAD50", "RAD51", "RAD51B", "RAD51C", "RAD51D", "RAD52", "SAMHD1", "SHFM1", "TP53", "TP53BP1", "WEE1", "WRN")


LRP_class_file <-  paste0(LRP_result_path,"/predicted_concordance.csv")
LRP_data <- paste0(LRP_result_path,"/relevances_rendered_class.csv")
# ML_feature_importance_path <- paste0(LRP_result_path,"/ML_feature_importance.csv")

CancerGeneCensus_file <- "./data/Cosmic_CancerGeneCensus_v98_GRCh37.tsv.gz"

# ML_feature_importanceDF <- read.csv(ML_feature_importance_path,header = T,check.names = F) 


##################
Get_top_LRP_genes <- function(LRP_data,LRP_class_file,top_n=NA){
  LRP_DF <- read.csv(LRP_data,header = T,check.names = F) 
  LRP_class_DF <- read.csv(LRP_class_file,header = T,check.names = F)
  if (is.na(top_n)) {
    top_n <- ncol(LRP_DF) - 1
  }
  
  Keep_PatientID <- LRP_class_DF[LRP_class_DF$Concordance == 1, "Patient ID"]
  LRP_DF <- LRP_DF %>% 
    dplyr::filter(`Patient ID` %in% Keep_PatientID)
  
  # Calculate mean of absolute LRP scores
  lrp_mean_scores <- LRP_DF %>% 
    dplyr::select(-`Patient ID`) %>%
    summarise(across(everything(), ~mean(abs(.), na.rm = TRUE))) %>%
    pivot_longer(cols = everything(), names_to = "Gene", values_to = "MeanAbsLRPScore") %>%
    arrange(desc(MeanAbsLRPScore)) %>%
    slice_head(n = top_n)
  
  # lrp_mean_scores <- LRP_DF %>% 
  #   dplyr::filter(`Patient ID` %in% Keep_PatientID) %>%
  #   dplyr::select(-`Patient ID`) %>%  # Exclude 'Patient_ID' from the operation
  #   summarise_all(~mean(abs(.))) 
  # top_100_genes <- lrp_mean_scores %>%
  #   pivot_longer(cols = everything(), names_to = "Gene", values_to = "MeanAbsLRPScore") %>%
  #   arrange(desc(MeanAbsLRPScore)) %>%
  #   slice(1:top_n)
  lrp_mean_scores
}

Top_100Genes_DF <- Get_top_LRP_genes(LRP_data,LRP_class_file,100)
Top_LRPGenes_DF <- Get_top_LRP_genes(LRP_data,LRP_class_file)

Top100_LRP_Genes_path<- file.path("./results/Top100_LRP_Genes.csv")
# write.table(Top_100Genes_DF,Top100_LRP_Genes_path, sep = ",",row.names = FALSE,col.names = TRUE)

# feature_importance_df_normalized <- ML_feature_importanceDF %>%
#   left_join(Top_LRPGenes_DF) %>%
#   dplyr::rename(GLRP = MeanAbsLRPScore) %>%
#   mutate(across(-Gene, ~ .x / sum(.x)))
# 
# feature_importance_df_normalizedLong <- feature_importance_df_normalized %>%
#   pivot_longer(cols = -Gene, names_to = "Model", values_to = "Importance")
# 
# # Rank genes within each model and select the top 100
# top_genes_per_model <- feature_importance_df_normalizedLong %>%
#   group_by(Model) %>%
#   mutate(Rank = rank(-Importance)) %>%
#   filter(Rank <= 100) %>%
#   ungroup() %>% 
#   arrange(Model, Rank)
# 
# top_genes_list <- split(top_genes_per_model, top_genes_per_model$Model)
# DT_topGenes <- top_genes_list$DecisionTree$Gene
# LR_topGenes <- top_genes_list$LogisticRegression$Gene
# SVC_topGenes <- top_genes_list$SVC$Gene
# GLRP_topGenes <- top_genes_list$GLRP$Gene
# 
# 
# RF_topGenes <- top_genes_list$RandomForest$Gene
# intersect(GLRP_topGenes,BRCA_genes)

#########################
########## Cancer gene and Transreg genes
### get the GO_term genes
go_term <- "GO:0006355"   ## "Regulation of transcription, DNA-templated"
go_term <- "GO:0003700"   ## "DNA-binding transcription factor activity"
go_term <- "GO:0140110"   ## "Transcription regulator activity"
GO_gene_retrieve <- function(go_term){
  genes <- as.character(org.Hs.egGO2ALLEGS[[go_term]])
  
  gene_symbols <- mapIds(org.Hs.eg.db, keys = genes, column = "SYMBOL", keytype = "ENTREZID")
  return(gene_symbols)
}
trans_reg_genes <- GO_gene_retrieve("GO:0006355")


Cancer_gene_retrieve <- function(CancerGeneCensus_file){
  CancerGeneCensus_DF <-as.data.frame(data.table::fread(CancerGeneCensus_file,sep = "\t",check.names = F,header=T,stringsAsFactors = F))
  cancer_genes <- unique(CancerGeneCensus_DF$GENE_SYMBOL)
  cancer_genes
}
cancer_genes <- Cancer_gene_retrieve(CancerGeneCensus_file)

pathway_name <- "R-HSA-5685942"   ## reactome HRR pathway ID
pathway_name <- "hsa03440"       ## KEGG HRR pathway ID
Get_HR_PathwayGenes <- function(pathway_name){
  
  if(startsWith(pathway_name, "R-HSA") ){  #### Reactome pathway name start with "R-HSA"
    pathways <- as.list(reactomePATHID2EXTID)
    # pathway_name <- "R-HSA-5685942"
    pathway_id <- names(pathways)[grep(pathway_name, names(pathways))]
    reactome_genes <- unlist(pathways[pathway_id])
    reactome_genes_hugo <- mapIds(org.Hs.eg.db, keys = reactome_genes, column = "SYMBOL", keytype = "ENTREZID")
    HR_PathwayGenes <- as.character(reactome_genes_hugo)
    return(HR_PathwayGenes)
  } else if(startsWith(pathway_name, "hsa")){                      #### KEGG pathway name start with "hsa"
    #Get the list of numbers, gene symbols and gene description
    gene_info <- keggGet(pathway_name)[[1]]$GENE
    #Delete the gene number by deleting every other line
    gene_info_odd <-  gene_info[seq(0,length(gene_info),2)]
    # gene_info_odd <- gene_info[seq(1, length(gene_info), 2)]
    #Create a substring deleting everything after the ; on each line (this deletes the gene description).
    HR_PathwayGenes <- gsub("\\;.*","",gene_info_odd)
    return(HR_PathwayGenes)
  } else{
    return(NULL)
  }
  
}
homologous_recombination_genes <- Get_HR_PathwayGenes("R-HSA-5685942") ##HRR
homologous_recombination_genes <- Get_HR_PathwayGenes("hsa03440")
FA_genes <- Get_HR_PathwayGenes("R-HSA-6783310")  ##FA

Candidate_pathways <- list(HRR = "R-HSA-5685942", FanconiAnemia_Pathway = "R-HSA-6783310",DSBsResponse = "R-HSA-5693606",
                           DSBsRepair = "R-HSA-5693532", Base_Excision_Repair = "R-HSA-73884"
                           , ESR_mediated_signaling = "R-HSA-8939211"
                           )


pathway_gene_list<- lapply(Candidate_pathways,Get_HR_PathwayGenes)

#######
#### Pathway/GO term enrichment analysis
top_genes <-  head(Top_100Genes_DF$Gene,100)
PathwayResult_Output_dir <- "./figures/Heatmap_Network_result"
find_pathways <- function(top_genes,Output_dir,output_filename) {
  if(missing(output_filename))
    output_filename = "TopGenes"
  
  # Convert gene symbols to Entrez IDs
  gene_ids <- bitr(top_genes, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)
  
  # Perform over-representation analysis
  enrich_result <- enrichPathway(gene = gene_ids$ENTREZID,
                                 pvalueCutoff = 0.05)
  
  # Convert Entrez IDs in enrichResult to gene symbols
  gene_symbols <- bitr(gsub("/"," ",enrich_result@gene), 
                       fromType = "ENTREZID", toType = "SYMBOL", OrgDb = org.Hs.eg.db)
  gene_map <- setNames(gene_symbols$SYMBOL, gene_symbols$ENTREZID)
  # Replace the Entrez IDs in the 'geneID' column with gene symbols
  enrich_result@result$GeneSymbols <- sapply(strsplit(as.character(enrich_result@result$geneID), "/"), function(x) {
    paste(gene_map[x], collapse = "/")
  })
  
  ### Get the GenePathwayRatio
  enrich_result@result$GeneNumerator <- sapply(strsplit(enrich_result@result$GeneRatio, "/"), `[`, 1)
  enrich_result@result$BgNumerator <- sapply(strsplit(enrich_result@result$BgRatio, "/"), `[`, 1)
  enrich_result@result$GenePathwayRatio <- paste(enrich_result@result$GeneNumerator, enrich_result@result$BgNumerator, sep="/")
  enrich_result@result$GenePathwayFreq <- strtoi(enrich_result@result$GeneNumerator)/strtoi(enrich_result@result$BgNumerator)
  enrich_result@result$GeneNumerator <- NULL
  enrich_result@result$BgNumerator <- NULL

  ### Get the enrichment df with p.adjust < 0.05
  enrich_result@result <-  enrich_result@result[enrich_result@result$p.adjust < 0.05,]
  if (!dir.exists(Output_dir)) {
    dir.create(Output_dir, recursive = TRUE)
    cat("Directory created at:", Output_dir, "\n")
  } else {
    cat("Directory already exists:", Output_dir, "\n")
  }
  
  Pathway_result_path <- file.path(Output_dir,paste0(output_filename, "_Pathway_result.csv"))
  write.table(enrich_result@result,Pathway_result_path, sep = ",",row.names = FALSE,col.names = TRUE)

  # Return the results
  return(enrich_result@result)
}
res <-find_pathways(top_genes,PathwayResult_Output_dir)

find_GO_terms <- function(top_genes, Output_dir,output_filename) {
  if(missing(output_filename))
    output_filename = "TopGenes"
  # Convert gene symbols to Entrez IDs
  gene_ids <- bitr(top_genes, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)
  
  # Perform GO enrichment analysis
  enrich_result <- enrichGO(gene = gene_ids$ENTREZID,
                            OrgDb = org.Hs.eg.db,
                            keyType = "ENTREZID",
                            ont = "BP",  # change "BP"(biological process) to "CC" or "MF" for Cellular Component or Molecular Function, respectively
                            pAdjustMethod = "BH",
                            pvalueCutoff = 0.05,
                            qvalueCutoff = 0.2)
  
  # Convert Entrez IDs in enrichResult to gene symbols
  gene_symbols <- bitr(gsub("/"," ",enrich_result@gene), 
                       fromType = "ENTREZID", toType = "SYMBOL", OrgDb = org.Hs.eg.db)
  gene_map <- setNames(gene_symbols$SYMBOL, gene_symbols$ENTREZID)
  
  # Replace the Entrez IDs in the 'geneID' column with gene symbols
  enrich_result@result$GeneSymbols <- sapply(strsplit(as.character(enrich_result@result$geneID), "/"), function(x) {
    paste(gene_map[x], collapse = "/")
  })
  
  ### Get the GenePathwayRatio
  enrich_result@result$GeneNumerator <- sapply(strsplit(enrich_result@result$GeneRatio, "/"), `[`, 1)
  enrich_result@result$BgNumerator <- sapply(strsplit(enrich_result@result$BgRatio, "/"), `[`, 1)
  enrich_result@result$GenePathwayRatio <- paste(enrich_result@result$GeneNumerator, enrich_result@result$BgNumerator, sep="/")
  enrich_result@result$GenePathwayFreq <- strtoi(enrich_result@result$GeneNumerator)/strtoi(enrich_result@result$BgNumerator)
  enrich_result@result$GeneNumerator <- NULL
  enrich_result@result$BgNumerator <- NULL
  
  ### Get the enrichment df with p.adjust < 0.05
  # enrich_result@result <-  enrich_result@result[enrich_result@result$p.adjust < 0.05,]
  if (!dir.exists(Output_dir)) {
    dir.create(Output_dir, recursive = TRUE)
    cat("Directory created at:", Output_dir, "\n")
  } else {
    cat("Directory already exists:", Output_dir, "\n")
  }
  
  GO_result_path <- file.path(Output_dir,paste0(output_filename, "_GO_result.csv"))
  write.table(enrich_result@result, GO_result_path, sep = ",", row.names = FALSE, col.names = TRUE)
  
  # Return the results
  return(enrich_result@result)
}
# Output_dir <- "/sybig/home/jiy/Downloads/BRCAness_project/Result/TCGA_expected_ALL"
res_GO <-find_GO_terms(top_genes,PathwayResult_Output_dir)
############################
