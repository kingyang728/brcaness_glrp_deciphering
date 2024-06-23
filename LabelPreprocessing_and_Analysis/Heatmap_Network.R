library(RColorBrewer)
library(ComplexHeatmap)
library(circlize)
library(igraph)
library(ggplot2)
# TOP_genes <- RF_topGenes

Top100_LRP_Genes_path = "./results/Top100_LRP_Genes.csv"
Top_100Genes_DF <-  data.table::fread(Top100_LRP_Genes_path,sep = ",",check.names = F,header=T,stringsAsFactors = F)
TOP_LRP_genes <- head(Top_100Genes_DF$Gene,100)
BRCAness_Label <- data.table::fread(BRCAness_Label_path,sep = ",",check.names = F,header=T,stringsAsFactors = F)
TCGA_PanCan_EXP <- data.table::fread(TCGA_PanCan_EXP_path,sep = ",",check.names = F,header=T,stringsAsFactors = F)
Clinical_file <- "./data/Survival_SupplementalTable_S1_20171025_xena_sp"
Clinical_DF <- data.table::fread(Clinical_file,sep = "\t",check.names = F,header=T,stringsAsFactors = F)
TOP_genes <- TOP_LRP_genes
Candidate_pathways <- list(HRR = "R-HSA-5685942", Fanconi_Anemia_Pathway = "R-HSA-6783310",DSBs_Response = "R-HSA-5693606",
                           DSBs_Repair = "R-HSA-5693532", Base_Excision_Repair = "R-HSA-73884",
                           Single_Strand_Annealing = "R-HSA-5685938", Nonhomologous_End_Joining = "R-HSA-5693571",
                           `MMEJ(alt_NHEJ)` = "R-HSA-5685939",
                           Homology_Directed_Repair = "R-HSA-5693538", 
                           # Cell_Cycle_CheckPoints = "R-HSA-69620",
                           ESR_mediated_signaling = "R-HSA-8939211"
)
pathway_gene_list<- lapply(Candidate_pathways,Get_HR_PathwayGenes)
BRCA_genes<-c("CCNE1","PALB2","ATM", "ATR", "AURKA", "BAP1", "BARD1", "BLM", "BRCA1", "BRCA2", "BRIP1", "CDK12", "CHD4", "CHEK1", "CHEK2", "C11orf30", "ERCC1", "FANCA", "FANCC", "FANCD2", "FANCE", "FANCF",
              "FANCI", "KMT2A", "MRE11A", "MYC", "NBN", "PALB2", "PARP1", "PAXIP1", "PLK1", "PTEN", "RAD50", "RAD51", "RAD51B", "RAD51C", "RAD51D", "RAD52", "SAMHD1", "SHFM1", "TP53", "TP53BP1", "WEE1", "WRN")

pathway_gene_list <- c(pathway_gene_list, list(BRCAness_genes = BRCA_genes))


# Get BRCAness_DF TCGA_PanCan_EXP & Top_gene_DF dataframe
DataFrame_Preprocess <- function(BRCAness_DF,TCGA_PanCan_EXP,TOP_genes,Clinical_DF){
  BRCAness_DF <-as.data.frame(t(BRCAness_Label))
  TCGA_PanCan_EXP <- as.data.frame(TCGA_PanCan_EXP)
  Top_gene_DF <- data.frame(
    probe = TOP_genes,
    rank = 1:length(TOP_genes)
  )
  CancerSample_DF <- Clinical_DF %>% dplyr::select(sample,`cancer type abbreviation`) %>% filter(sample %in% colnames(TCGA_PanCan_EXP))
  list(TCGA_PanCan_EXP = TCGA_PanCan_EXP, BRCAness_DF = BRCAness_DF,Top_gene_DF = Top_gene_DF,CancerSample_DF = CancerSample_DF)
  
}
# DataFrame_list <- DataFrame_Preprocess(BRCAness_DF,TCGA_PanCan_EXP,TOP_genes)
# TCGA_PanCan_EXP <- DataFrame_list$TCGA_PanCan_EXP
# BRCAness_DF <- DataFrame_list$BRCAness_DF

## Create output Dir
create__HeatmapNetwork_OutputDir <- function(file_path, new_dir_name) {
  # Extract the base path (up to and including "analysis/")
  base_path <- dirname(dirname(dirname(file_path)))
  
  # Create the full path for the new directory
  new_dir_path <- file.path(base_path, new_dir_name)
  
  # Check if the directory exists and create it if it doesn't
  if (!dir.exists(new_dir_path)) {
    dir.create(new_dir_path, recursive = TRUE)
    cat("Directory created at:", new_dir_path, "\n")
  } else {
    cat("Directory already exists:", new_dir_path, "\n")
  }
  new_dir_path
}

Get_CancerType_GCNNPerformance_results <- function(Clinical_DF,LRP_class_file,TCGA_PanCan_EXP,CancerSample_DF){
  # CancerSample_DF <- Clinical_DF %>% dplyr::select(sample,`cancer type abbreviation`) %>% filter(sample %in% colnames(TCGA_PanCan_EXP))
  LRP_class_DF <- read.csv(LRP_class_file,header = T,check.names = F)
  ## remove the cancer type which have only one class (BRCAness/Non-BRCAness)
  LRP_classCancerType_DF <- CancerSample_DF %>% inner_join(LRP_class_DF,  by=c("sample" = "Patient ID")) %>%  
    group_by(`cancer type abbreviation`) %>% filter(sum(label == 0) >= 2 & sum(label == 1) >= 2) %>% ungroup() 
  ## Bar plot of sample counts in training
  LRP_classCancerType_PlotDF <- LRP_classCancerType_DF %>% dplyr::select(`cancer type abbreviation`,label) %>%  mutate(BRCAness =ifelse(label == 1, "BRCAness","Non_BRCAness")) %>% add_count(`cancer type abbreviation`,BRCAness) %>% distinct()
  Testset_cancerSubtype_plot <- ggplot(LRP_classCancerType_PlotDF, aes(`cancer type abbreviation`, n, fill = BRCAness)) + 
    geom_bar(stat="identity", position = "dodge") + 
    scale_fill_brewer(palette = "Set1") + geom_text(position = position_dodge2(width= 0.9),aes(label = n),size= 4,hjust=0.5, vjust= -0.5,fontface = "bold") +
    labs(x = "Cancer Type", y = "Count") + # Rename x-axis title to "Disease" and y-axis title to "Count"
    theme(axis.text.x = element_text(face = "bold",size = 12), 
          axis.text.y = element_text(face = "bold",size = 12),
          axis.title.x = element_text(face = "bold", size = 14),
          axis.title.y = element_text(face = "bold", size = 14))
  testSetSamples <- LRP_classCancerType_DF %>% pull(sample) %>% append("probe")
  CancerType_GCNNPerformance_results <- LRP_classCancerType_DF %>%
    group_by(`cancer type abbreviation`) %>%
    summarise(
      Accuracy = sum(Concordance) / n(),
      Recall = sum(Predicted == 1 & label == 1) / (sum(Predicted == 1 & label == 1) + sum(Predicted == 0 & label == 1)),
      Precision = sum(Predicted == 1 & label == 1) / (sum(Predicted == 1 & label == 1) + sum(Predicted == 1 & label == 0)),
      F1_Score = 2 * ((Precision * Recall) / (Precision + Recall))
    )
  cancer_names <- data.frame(
    abbreviation = c("ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM", "HNSC", "KICH", "KIRC", 
                     "KIRP", "LGG", "LIHC", "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", 
                     "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM"),
    full_name = c("Adrenocortical carcinoma", "Bladder urothelial carcinoma", "Breast invasive carcinoma", 
                  "Cervical squamous cell carcinoma and endocervical adenocarcinoma", "Cholangiocarcinoma", 
                  "Colon adenocarcinoma", "Lymphoid Neoplasm Diffuse Large B-cell Lymphoma", "Esophageal carcinoma", 
                  "Glioblastoma multiforme", "Head and Neck squamous cell carcinoma", "Kidney Chromophobe", 
                  "Kidney renal clear cell carcinoma", "Kidney renal papillary cell carcinoma", "Brain Lower Grade Glioma", 
                  "Liver hepatocellular carcinoma", "Lung adenocarcinoma", "Lung squamous cell carcinoma", 
                  "Mesothelioma", "Ovarian serous cystadenocarcinoma", "Pancreatic adenocarcinoma", 
                  "Pheochromocytoma and Paraganglioma", "Prostate adenocarcinoma", "Rectum adenocarcinoma", 
                  "Sarcoma", "Skin Cutaneous Melanoma", "Stomach adenocarcinoma", "Testicular Germ Cell Tumors", 
                  "Thyroid carcinoma", "Thymoma", "Uterine Corpus Endometrial Carcinoma", "Uterine Carcinosarcoma", 
                  "Uveal Melanoma")
  )
  CancerType_GCNNPerformance_results <- merge(CancerType_GCNNPerformance_results, cancer_names, by.x="cancer type abbreviation",by.y = "abbreviation", all=TRUE)

  
  TCGA_PanCan_Testset_EXPKeep <- TCGA_PanCan_EXP[TCGA_PanCan_EXP$probe %in% TOP_genes,testSetSamples]
  list(CancerType_GCNNPerformance_results = CancerType_GCNNPerformance_results, TCGA_PanCan_Testset_EXPKeep = TCGA_PanCan_Testset_EXPKeep,Testset_cancerSubtype_plot = Testset_cancerSubtype_plot)
  
}
# CancerType_GCNNPerformance_results_list <- Get_CancerType_GCNNPerformance_results(Clinical_DF,LRP_class_file,DataFrame_list$TCGA_PanCan_EXP)
# CancerType_GCNNPerformance_results<- CancerType_GCNNPerformance_results_list$CancerType_GCNNPerformance_results
# TCGA_PanCan_EXPKeep <- TCGA_PanCan_EXP[TCGA_PanCan_EXP$probe %in% TOP_genes,]


TCGA_PanCan_EXPKeep_ListCreate <- function(TCGA_PanCan_EXPKeep,CancerSample_DF){
  TCGA_PanCan_EXPKeep_tmp <- as.data.frame(t(TCGA_PanCan_EXPKeep[,!names(TCGA_PanCan_EXPKeep) %in% c("probe")]))
  colnames(TCGA_PanCan_EXPKeep_tmp) <- TCGA_PanCan_EXPKeep$probe
  TCGA_PanCan_EXPKeep_tmp$sample <- rownames(TCGA_PanCan_EXPKeep_tmp)
  TCGA_PanCan_EXPKeep_tmpList <- CancerSample_DF %>% right_join(TCGA_PanCan_EXPKeep_tmp) %>% group_by(`cancer type abbreviation`) %>% 
    split(f = as.factor(.$`cancer type abbreviation`)) 
  
  TCGA_PanCan_EXPKeep_tmpList <- lapply(TCGA_PanCan_EXPKeep_tmpList, function(x) {
    x$`cancer type abbreviation` <- NULL
    # rownames(x) <- x$sample
    # x$sample <-NULL
    return(x)
  })
  TCGA_PanCan_EXPKeep_tmpList$ALL <- TCGA_PanCan_EXPKeep_tmp
  TCGA_PanCan_EXPKeep_tmpList
}
# TCGA_PanCan_EXPKeep_tmpList <- TCGA_PanCan_EXPKeep_ListCreate(TCGA_PanCan_EXPKeep)
# if(exists("TCGA_PanCan_Testset_EXPKeep")){
#   TCGA_PanCan_EXPKeep_testSetList <- TCGA_PanCan_EXPKeep_ListCreate(TCGA_PanCan_Testset_EXPKeep)
# }


# BRCAness_DF_ALL <- BRCAness_DF %>% dplyr::rename(BRCAness= 1) %>% tibble::rownames_to_column( "sample") %>% mutate(BRCAlabel = ifelse(BRCAness==1,"BRCAness","NonBRCAness")) %>%
#   arrange(BRCAlabel) %>% left_join(Clinical_DF) %>% dplyr::rename(`Cancer Type`= `cancer type abbreviation`)

Get_resampled_Barplot <- function(BRCAness_DF_ALL){
  ## Bar plot of sample counts in training
  BRCAness_DF_plot <- BRCAness_DF_ALL %>% dplyr::select(`Cancer Type`,BRCAness) %>%  mutate(BRCAness =ifelse(BRCAness == 1, "BRCAness","Non_BRCAness")) %>% add_count(`Cancer Type`,BRCAness) %>% distinct()
  resampled_cancerSubtype_plot <- ggplot(BRCAness_DF_plot, aes(`Cancer Type`, n, fill = BRCAness)) + 
    geom_bar(stat="identity", position = "dodge") + 
    scale_fill_brewer(palette = "Set1") + geom_text(position = position_dodge2(width= 0.9),aes(label = n),size= 4,hjust=0.5, vjust= -0.5,fontface = "bold") +
    labs(x = "Cancer Type", y = "Count") + # Rename x-axis title to "Disease" and y-axis title to "Count"
    theme(axis.text.x = element_text(face = "bold",size = 12), 
          axis.text.y = element_text(face = "bold",size = 12),
          axis.title.x = element_text(face = "bold", size = 14),
          axis.title.y = element_text(face = "bold", size = 14))
  return(resampled_cancerSubtype_plot)
}
# resampled_cancerSubtype_plot <- Get_resampled_Barplot(BRCAness_DF_ALL)
# TCGA_PanCan_EXPKeep_tmpListDF <- TCGA_PanCan_EXPKeep_tmpList$BLCA
# TCGA_PanCan_EXPKeep_tmpList<- TCGA_PanCan_EXPKeep_testSetList
# TCGA_PanCan_EXPKeep_CancerType_Name <- "BRCA"
CancerTypeDF_list_analysis <- function(TCGA_PanCan_EXPKeep_CancerType_Name, TCGA_PanCan_EXPKeep_tmpList,BRCAness_DF_ALL,Top_gene_DF,Output_Dir){
  # TCGA_PanCan_EXPKeep_tmpListDF <- TCGA_PanCan_EXPKeep_tmpListDF
  BRCAness_DF <- BRCAness_DF_ALL
  # TCGA_PanCan_EXPKeep_tmpListDF$`cancer type abbreviation` <- NULL
  TCGA_PanCan_EXPKeep_tmpListDF <- TCGA_PanCan_EXPKeep_tmpList[[TCGA_PanCan_EXPKeep_CancerType_Name]]
  Subtype_DF <- as.data.frame(t(TCGA_PanCan_EXPKeep_tmpListDF))
  colnames(Subtype_DF) <- Subtype_DF["sample",]
  Subtype_DF <- Subtype_DF[-which(rownames(Subtype_DF) == "sample"), ]
  
  
  Subtype_DF <- Subtype_DF %>% mutate_if(is.character, as.numeric)
  # Syntax
  # Subtype_DF$column = as.numeric(as.character(my_dataframe$column)) 
  
  Subtype_DF$probe <- rownames(Subtype_DF)
  expr <- Subtype_DF[,names(Subtype_DF) %in% c(unique(BRCAness_DF$sample),"probe")]
  
  BRCA_patientID<- intersect(names(Subtype_DF) ,BRCAness_DF[BRCAness_DF$BRCAness == 1,]$sample)
  NonBRCA_patientID<- intersect(names(Subtype_DF) , BRCAness_DF[BRCAness_DF$BRCAness == 0,]$sample)
  expr_BRCA <- expr[,BRCA_patientID]
  expr_NonBRCA <- expr[,c(NonBRCA_patientID)]
  
  
  
  Expr_df_reorder <- function(expr){
    column_means <- colMeans(expr)
    # Reorder columns based on descending means
    expr_pre <- expr[, order(column_means, decreasing = TRUE)]
    expr_pre
  }
  expr_BRCA <- Expr_df_reorder(expr_BRCA)
  expr_NonBRCA <- Expr_df_reorder(expr_NonBRCA)
  expr_pre <- cbind(expr_BRCA,expr_NonBRCA)
  BRCAness_DF <- BRCAness_DF[match(colnames(expr_pre), BRCAness_DF$sample),]   ## reorder the rows based on colnames of expr_pre
  expr_pre$probe <- expr$probe
  expr <- expr_pre
  
  # Create additional information for each gene
  expr$is_cancer_gene <- ifelse(expr$probe %in% cancer_genes, "Cancer", "NonCancer")
  expr$is_trans_reg_gene <- ifelse(expr$probe %in% trans_reg_genes, "TransReg", "NonTransReg")
  
  mat = as.matrix(expr[, grep("TCGA", colnames(expr))])
  mat_scaled = t(apply(mat, 1, scale))
  colnames(mat_scaled) <- colnames(mat)
  rownames(mat_scaled) <- expr$probe
  FisherTestTb_Create<-function(mat_scaled,testType){
    count_pos_neg <- function(x) {
      list(pos = sum(x > 0), neg = sum(x < 0))
    }
    pos_neg_DFCreate <- function(mat_scaled,BRCAlabel_Str){
      samples = BRCAness_DF[BRCAness_DF$BRCAlabel == BRCAlabel_Str,]$sample
      result_list <- apply(mat_scaled[,samples], 1, count_pos_neg)
      result_df <- as.data.frame(do.call(rbind, result_list))
      
      result_df$pos <- sapply(result_df$pos, unlist)  ## unlist
      result_df$neg <- sapply(result_df$neg, unlist)
      
      names(result_df)[names(result_df) == "pos"] <- paste0(BRCAlabel_Str, "_pos")
      names(result_df)[names(result_df) == "neg"] <- paste0(BRCAlabel_Str, "_neg")
      result_df
      
    }
    BRCA_result_Df <- pos_neg_DFCreate(mat_scaled,"BRCAness")
    NonBRCA_result_Df <- pos_neg_DFCreate(mat_scaled,"NonBRCAness")
    FisherTest_DF <- cbind(BRCA_result_Df,NonBRCA_result_Df)
    fisherTestResult <- apply(FisherTest_DF, 1,
                              function(x) {
                                tbl <- matrix(as.numeric(x[1:4]), ncol=2, byrow=F)
                                fisherTestResult <- fisher.test(tbl, alternative=testType) #### oneside test vs two side test, two.sided/greater/less
                                c(p.value=fisherTestResult$p.value , oddsRatio=as.numeric(fisherTestResult$estimate))
                              })
    fisherTestResult<-as.data.frame(t(fisherTestResult))
    FisherTest_DF<-cbind(FisherTest_DF,fisherTestResult)
    FisherTest_TB <- FisherTest_DF[FisherTest_DF$p.value< 0.05,]
    
  }
  FisherTest_TB_twosided <- FisherTestTb_Create(mat_scaled,"two.sided")
  FisherTest_TB_greater <- FisherTestTb_Create(mat_scaled,"greater")
  FisherTest_TB_less <- FisherTestTb_Create(mat_scaled,"less")
  FisherTest_TB_list <- list(FisherTest_TB_twosided = FisherTest_TB_twosided, FisherTest_TB_greater = FisherTest_TB_greater,FisherTest_TB_less = FisherTest_TB_less)
  MeanMedian_DFCreate <- function(mat_scaled){
    BRCA_sample = BRCAness_DF[BRCAness_DF$BRCAlabel == "BRCAness",]$sample
    NonBRCA_sample = BRCAness_DF[BRCAness_DF$BRCAlabel == "NonBRCAness",]$sample
    
    BRCA_result_df_med <- as.data.frame(apply(mat_scaled[,BRCA_sample], 1, median))
    NonBRCA_result_df_med <- as.data.frame(apply(mat_scaled[,NonBRCA_sample], 1, median))
    
    BRCA_result_df_mean <- as.data.frame(apply(mat_scaled[,BRCA_sample], 1, mean))
    NonBRCA_result_df_mean <- as.data.frame(apply(mat_scaled[,NonBRCA_sample], 1, mean))
    
    result_df <-dplyr::bind_cols(BRCA_result_df_med, NonBRCA_result_df_med,BRCA_result_df_mean,NonBRCA_result_df_mean)
    
    names(result_df) <- c("BRCA_med","NonBRCA_med","BRCA_mean","NonBRCA_mean")
    
    result_df
    
  }
  MeanMedian_DF <- MeanMedian_DFCreate(mat_scaled)
  
  ### Get TTest_df
  test_result_med <- t.test(MeanMedian_DF$BRCA_med , MeanMedian_DF$NonBRCA_med,paired=TRUE, var.equal= TRUE)
  test_result_mean <- t.test(MeanMedian_DF$BRCA_mean,MeanMedian_DF$NonBRCA_mean , paired=TRUE, var.equal= TRUE)
  TTest_df <- data.frame(
    statistic = c(test_result_med$statistic,test_result_mean$statistic),
    p.value =c(test_result_med$p.value,test_result_mean$p.value),
    estimate = c(test_result_med$estimate,test_result_mean$estimate),
    alternative = c(test_result_med$alternative,test_result_mean$alternative),
    data.name = c(test_result_med$data.name,test_result_mean$data.name)
  )
  
  ### Get WilcoxTest_df
  WilcoxTest_result_med_G <- wilcox.test(MeanMedian_DF$BRCA_med ,MeanMedian_DF$NonBRCA_med, alternative = "greater")
  WilcoxTest_result_med_L <- wilcox.test(MeanMedian_DF$BRCA_med ,MeanMedian_DF$NonBRCA_med, alternative = "less")
  WilcoxTest_result_med_T <- wilcox.test(MeanMedian_DF$BRCA_med ,MeanMedian_DF$NonBRCA_med, alternative = "two.sided")
  WilcoxTest_result_mean_G <- wilcox.test(MeanMedian_DF$BRCA_mean ,MeanMedian_DF$NonBRCA_mean, alternative = "greater")
  WilcoxTest_result_mean_L <- wilcox.test(MeanMedian_DF$BRCA_mean ,MeanMedian_DF$NonBRCA_mean, alternative = "less")
  WilcoxTest_result_mean_T <- wilcox.test(MeanMedian_DF$BRCA_mean ,MeanMedian_DF$NonBRCA_mean, alternative = "two.sided")
  WilcoxTest_df <- data.frame(
    p.value_greater = c(WilcoxTest_result_med_G$p.value,WilcoxTest_result_mean_G$p.value),
    p.value_less = c(WilcoxTest_result_med_L$p.value,WilcoxTest_result_mean_L$p.value),
    p.value_twoSide = c(WilcoxTest_result_med_T$p.value,WilcoxTest_result_mean_T$p.value),
    data.name = c(WilcoxTest_result_med_T$data.name,WilcoxTest_result_mean_T$data.name)
  )
  
  
  MeanMedian_DF_scaled = as.matrix(MeanMedian_DF)
  MeanMedian_DF$probe = rownames(MeanMedian_DF)
  MeanMedian_DF$Med_diff <- MeanMedian_DF$BRCA_med - MeanMedian_DF$NonBRCA_med
  MeanMedian_DF <- MeanMedian_DF %>% arrange(desc(Med_diff))
  expr <- expr %>% left_join(MeanMedian_DF,by = "probe") %>% arrange(desc(Med_diff)) %>% left_join(Top_gene_DF)
  
  typeDF <- BRCAness_DF[,c("BRCAlabel","Cancer Type","age_at_initial_pathologic_diagnosis","gender")]
  num_cancer_types <- length(unique(typeDF$`Cancer Type`))
  color_palette <- colorRampPalette(brewer.pal(12, "Set3"))(num_cancer_types)
  # color_palette <- rainbow(num_cancer_types)
  CancerType_color_mapping <- setNames(color_palette, unique(typeDF$`Cancer Type`))
  split = typeDF$`Cancer Type`
  column_ha = HeatmapAnnotation(
    # text = anno_text(typeDF$`Cancer Type`,rot = 45,gp = gpar(fontsize = 7)),
    `Cancer Type` = typeDF$`Cancer Type`,
    Gender = typeDF$gender,
    Age = typeDF$age_at_initial_pathologic_diagnosis,
    BRCAness = typeDF$BRCAlabel,
    col = list(BRCAness = c("BRCAness" =  "orange", "NonBRCAness" = "grey"),
               Gender = c("FEMALE" =  "pink", "MALE" = "cyan"),
               Age = colorRamp2(c( 20,50,80), c("white","grey","black")),
               `Cancer Type` = CancerType_color_mapping
    )
    
  )
  col_fun = colorRamp2(c(-2, 0, 2), c("blue", "white", "red"))
  rank_colors <- colorRamp2(c(  length(expr$rank),length(expr$rank)/2,1), c("white","lightskyblue1", "deepskyblue2"))
  row_ha = rowAnnotation(Cancer = expr$is_cancer_gene,TransReg = expr$is_trans_reg_gene,gene_rank = expr$rank,
                         col = list(Cancer = c("Cancer" = "red", "NonCancer" = "white"),TransReg = c("TransReg" = "darkgreen", "NonTransReg" = "white"),
                                    gene_rank = rank_colors),
                         width = unit(5, "mm")
  )
  
  Lrow_ha = rowAnnotation(
    brcaness_med = expr$BRCA_med, nonbrcaness_med  = expr$NonBRCA_med, 
    # brcaness_mean = expr$BRCA_mean, nonbrcaness_mean = expr$NonBRCA_mean, 
    col = list(
      brcaness_med = col_fun,nonbrcaness_med = col_fun
      #, brcaness_mean = col_fun,nonbrcaness_mean = col_fun 
    ),
    width = unit(1, "mm")
  )
  # Create a color vector for the gene names
  gene_colors = rep("black", length(expr$probe))
  gene_colors[expr$is_cancer_gene == "Cancer" & expr$is_trans_reg_gene == "NonTransReg"] <- "red"
    gene_colors[expr$is_cancer_gene == "NonCancer" & expr$is_trans_reg_gene == "TransReg"] <- "darkgreen"
      gene_colors[expr$is_cancer_gene == "Cancer" & expr$is_trans_reg_gene == "TransReg"] <- "purple"
        names(gene_colors) <- expr$probe
        mat_scaled <- mat_scaled[expr$probe, ]  #reorder the rows of mat_scaled.
        # BRCA_m_plot <- Heatmap(MeanMedian_DF_scaled, name = "expression",
        #                        col = colorRamp2(c(-2, 0, 2), c("green", "white", "red")),   row_order = MeanMedian_DF$probe,
        #                        column_order = colnames(MeanMedian_DF_scaled),show_row_names = T,  row_names_gp = gpar(col = gene_colors, fontsize = 8))
        BRCAness_p <- Heatmap(mat_scaled, name = "expression",
                              col = colorRamp2(c(-2, 0, 2), c("blue", "white", "red")),
                              top_annotation = column_ha,
                              show_row_names = T,
                              show_column_names = F,
                              # column_split = split_BRCA,
                              column_order = BRCAness_DF$sample,
                              row_order = expr$probe,
                              right_annotation =  row_ha,
                              left_annotation = Lrow_ha,
                              row_names_gp = gpar(col = gene_colors, fontsize = 8),
                              column_names_gp = gpar(fontsize = 8)
        )
        if(length(unique(split)) >1){
          CancerType_split_p <- Heatmap(mat_scaled, name = "expression",
                                        col = colorRamp2(c(-2, 0, 2), c("blue", "white", "red")),
                                        column_split = split,
                                        top_annotation = column_ha,
                                        column_title_rot = 45,
                                        show_row_names = T,
                                        show_column_names = F,
                                        column_order = BRCAness_DF$sample,
                                        row_order = expr$probe,
                                        right_annotation =  row_ha,
                                        left_annotation = Lrow_ha,
                                        row_names_gp = gpar(col = gene_colors, fontsize = 8),
                                        column_names_gp = gpar(fontsize = 8)
          )
        } else {
          CancerType_split_p <- NULL
          
        }
        
        plot_top_genesDiffNetwork <- function(reactome_FIs_TB, MeanMedian_DF,pathway_gene_list) {
          MeanMedian_DF$cancer_related <- ifelse(MeanMedian_DF$probe %in% cancer_genes, "Yes", "No")
          MeanMedian_DF$trans_reg <- ifelse(MeanMedian_DF$probe  %in% trans_reg_genes, "Yes", "No")
          rownames(reactome_FIs_TB) <- colnames(reactome_FIs_TB) 
          sub_matrix <- reactome_FIs_TB[rownames(reactome_FIs_TB) %in% MeanMedian_DF$probe,
                                        colnames(reactome_FIs_TB) %in% MeanMedian_DF$probe]
          g <- graph_from_adjacency_matrix(sub_matrix, mode = "undirected", weighted = NULL, diag = FALSE, add.colnames = NULL)
          cl <- clusters(g)
          small.clusters <- which(cl$csize <= 1)
          vertices.to.delete <- which(cl$membership %in% small.clusters)
          g <- delete.vertices(g, vertices.to.delete)   ## remove the isolate vertices
          
          Pathwaygroup_ids <- lapply(pathway_gene_list, function(pathway_genes) {
            intersect(pathway_genes, V(g)$name)
          })
          # Filter out empty lists
          Pathwaygroup_ids <- Pathwaygroup_ids[sapply(Pathwaygroup_ids, length) > 0]
          
          ## function to merge the list element which hold the same vector.
          mergeListBasedOnContent <- function(lst) {
            # Create a list to store merged names
            merged_names <- list()
            
            # Iterate over each list element and create a sorted string representation
            for(name in names(lst)) {
              key <- paste(sort(lst[[name]]), collapse = "-")
              if(!is.null(merged_names[[key]])) {
                # If the key already exists, append the new name to it
                merged_names[[key]] <- paste(merged_names[[key]], name, sep = "/")
              } else {
                # Otherwise, create a new entry
                merged_names[[key]] <- name
              }
            }
            
            # Create the new merged list based on the stored names
            merged_lst <- list()
            for(key in names(merged_names)) {
              merged_lst[[merged_names[[key]]]] <- lst[[unlist(strsplit(merged_names[[key]], "/"))[1]]]
            }
            
            return(merged_lst)
          }  
          Pathwaygroup_ids <- mergeListBasedOnContent(Pathwaygroup_ids)
          # group_color_fill <- colorRampPalette(brewer.pal(12, "Set3"))(length(Pathwaygroup_ids))
          group_color_fill <- rainbow(length(Pathwaygroup_ids))
          group_color_fill_transparent <- adjustcolor(group_color_fill, alpha.f = 0.2)
          
          group_color_border <- group_color_fill # This could be different if desired
          
          color_map <- circlize::colorRamp2(c(-2, 0, 2), c("blue", "white", "red"))
          
          LO_Customized = layout_with_fr(g)
          # Split plotting area into 1 row and 2 columns
          par(mfrow = c(1, 1),oma=c(0,0,0,4), mar = c(5,4,4,6) + 0.1)
          V(g)$mean_exp_diff <- MeanMedian_DF$BRCA_med[match(V(g)$name, MeanMedian_DF$probe)] - MeanMedian_DF$NonBRCA_med[match(V(g)$name, MeanMedian_DF$probe)]
          V(g)$color <- color_map(V(g)$mean_exp_diff)
          V(g)$label.color <- ifelse(MeanMedian_DF$cancer_related[match(V(g)$name, MeanMedian_DF$probe)] == "Yes", "black", "black")
          V(g)$frame.color <- ifelse(
            MeanMedian_DF$cancer_related[match(V(g)$name, MeanMedian_DF$probe)] == "Yes",
            adjustcolor("red", alpha.f = 0.4),
            adjustcolor("black", alpha.f = 0.4)
          )
          V(g)$shape <- ifelse(MeanMedian_DF$trans_reg[match(V(g)$name, MeanMedian_DF$probe)] == "Yes", "square", "circle")
          gene_title <- paste("Expression difference plot of top", paste(length(TOP_genes), collapse = ","),"GLRP genes between BRCAness & Non-BRCAness (",TCGA_PanCan_EXPKeep_CancerType_Name,")")
          plot(g, layout = LO_Customized,
               vertex.label=V(g)$name,
               vertex.label.font = 2, # Font:  1 plain, 2 bold, 3, italic, 4 bold italic
               vertex.size=5, 
               vertex.label.cex = 0.7,
               vertex.color=V(g)$color ,
               vertex.frame.color=V(g)$frame.color,
               vertex.label.color=V(g)$label.color,
               mark.groups = Pathwaygroup_ids, 
               mark.col = group_color_fill_transparent, 
               mark.border = group_color_border,
               main=gene_title)
          formatted_legend <- sapply(names(Pathwaygroup_ids), function(name) {
            
            # Original code for splitting genes remains the same
            split_string <- function(string, max_length) {
              words <- unlist(strsplit(string, split=","))
              output <- c()
              line <- words[1]
              
              for (word in words[-1]) {
                if (nchar(line) + nchar(word) + 1 <= max_length) {
                  line <- paste(line, word, sep=",")
                } else {
                  output <- c(output, line)
                  line <- word
                }
              }
              output <- c(output, line)
              
              return(paste(output, collapse="\n"))
            }
            
            genes <- paste(Pathwaygroup_ids[[name]], collapse = ", ")
            genes <- split_string(genes, 30)
            
            # Function to split pathway names to ensure each line doesn't exceed max_length
            split_pathway_name <- function(pathway_name, max_length) {
              split_positions <- c()
              current_pos <- 1
              
              while (current_pos + max_length <= nchar(pathway_name)) {
                # Find the position closest to max_length characters to split
                split_pos <- current_pos + max_length
                while (split_pos > current_pos && substr(pathway_name, split_pos, split_pos) != "/") {
                  split_pos <- split_pos - 1
                }
                
                # If a suitable split position is found, record it and adjust current_pos
                if (split_pos > current_pos) {
                  split_positions <- c(split_positions, split_pos)
                  current_pos <- split_pos + 1
                } else {
                  # If no "/" found within the limit, force a split at max_length (or at the end of the string if shorter)
                  current_pos <- current_pos + max_length
                }
              }
              
              # Split the pathway name at recorded positions
              if (length(split_positions) > 0) {
                pathway_name_split <- ""
                start_pos <- 1
                for (pos in split_positions) {
                  pathway_name_split <- paste(pathway_name_split, substr(pathway_name, start_pos, pos), sep="")
                  start_pos <- pos + 1
                  if (start_pos < nchar(pathway_name)) {
                    pathway_name_split <- paste(pathway_name_split, "\n", sep="")
                  }
                }
                # Add the remaining part of the pathway name
                pathway_name_split <- paste(pathway_name_split, substr(pathway_name, start_pos, nchar(pathway_name)), sep="")
                return(pathway_name_split)
              } else {
                return(pathway_name)
              }
            }
            
            # Apply function to split long pathway names
            name <- split_pathway_name(name, 30) # Assuming 30 characters before splitting
            
      
            return(paste(name, genes, sep = ":\n"))
          })
        
          # formatted_legend <- sapply(names(Pathwaygroup_ids), function(name) {
          #   split_string <- function(string, max_length) {
          #     words <- unlist(strsplit(string, split=","))
          #     output <- c()
          #     line <- words[1]
          # 
          #     for (word in words[-1]) {
          #       if (nchar(line) + nchar(word) + 1 <= max_length) {
          #         line <- paste(line, word, sep=",")
          #       } else {
          #         output <- c(output, line)
          #         line <- word
          #       }
          #     }
          #     output <- c(output, line)
          # 
          #     return(paste(output, collapse="\n"))
          #   }
          #   genes <- paste(Pathwaygroup_ids[[name]], collapse = ", ")
          #   genes <- split_string(genes, 30)
          #   return(paste(name, genes, sep = ":\n"))
          # })
          
          legend('left', legend = formatted_legend,
                 col = group_color_fill_transparent,
                 pch = 15, bty = "n",  pt.cex = 1.5, cex = 0.8, 
                 text.col = "black", horiz = FALSE, y.intersp = 1.5, xpd = TRUE)
          # Restore default plotting parameters
          par(mfrow = c(1, 1),oma=c(0,0,0,0))
          legend_size = 0.8 
          
          # Color legend
          
          # Shape legend
          legend(x = "topright",
                 legend = c("Transcription Regulator", "No Transcription Regulator"),
                 pch = c(0, 1),  # 0 for square, 1 for circle
                 cex = legend_size,
                 title = "TR Status")
          legend(x = "right",
                 legend = c("High (brcaness > non-brcaness)", "Median (brcaness = non-brcaness)", "Low (brcaness < non-brcaness)"),
                 fill = c("red", "white", "blue"),
                 cex = legend_size,
                 title = "Mean expression difference \nbetween BRCAness/Non-BRCAness")
          
          # Label color legend
          legend(x = "bottomright",
                 legend = c("Cancer Related", "No Cancer Related"),
                 pch = c(0, 0),              # Point type 21 is a filled circle, you can adjust as per your plot
                 
                 pt.cex = 1,                 # Point size
                 col = c("red", "black"),      # Point border color
                 # text.col = c("red", "black"), 
                 cex = legend_size,
                 title.col = "black",
                 title = "Cancer Gene",
                 bty = "o")                    # Box type "o" will draw a box around the legend
          p <- recordPlot()
          return(p)
        }
      
        ##  output plot
        heatmap_filename <- paste0(TCGA_PanCan_EXPKeep_CancerType_Name,"_Heatmap.png")
        split_heatmap_filename <- paste0(TCGA_PanCan_EXPKeep_CancerType_Name,"_CancerType_Split_Heatmap.png")
        network_plot_filename <- paste0(TCGA_PanCan_EXPKeep_CancerType_Name,"_Network_Plot.png")
        TTest_df_filename <- paste0(TCGA_PanCan_EXPKeep_CancerType_Name,"_TTest_df.csv")
        WilcoxTest_df_filename <- paste0(TCGA_PanCan_EXPKeep_CancerType_Name,"_WilcoxTest_df.csv")
        MeanMedian_DF_filename <- paste0(TCGA_PanCan_EXPKeep_CancerType_Name,"_MeanMedian_DF.csv")

        # Start PNG device for BRCAness_p
        png(file.path(Output_Dir, heatmap_filename), width = 6144, height = 3840, res = 300)
        # Draw the heatmap
        draw(BRCAness_p)
        # Close the device
        dev.off()
        if (!is.null(CancerType_split_p)) {
          # Start PNG device for CancerType_split_p
          png(file.path(Output_Dir, split_heatmap_filename), width = 6144, height = 3840, res = 300)
          # Draw the heatmap
          draw(CancerType_split_p)
          # Close the device
          dev.off()
        }
        # Start PNG device for Network_Plot
        png(file.path(Output_Dir, network_plot_filename), width = 6144, height = 3840, res = 300)
        # Network plot
        Network_Plot <- plot_top_genesDiffNetwork(reactome_FIs_TB, MeanMedian_DF, pathway_gene_list)
        # Close the device
        dev.off()
        write.csv(TTest_df,  file.path(Output_Dir, TTest_df_filename) , row.names = FALSE, na="") 
        write.csv(WilcoxTest_df,  file.path(Output_Dir, WilcoxTest_df_filename) , row.names = FALSE, na="") 
        write.csv(MeanMedian_DF,  file.path(Output_Dir, MeanMedian_DF_filename) , row.names = FALSE, na="") 
        list(BRCAness_p = BRCAness_p,FisherTest_TB_list = FisherTest_TB_list,TTest_df = TTest_df, WilcoxTest_df = WilcoxTest_df, CancerType_split_p = CancerType_split_p, Network_Plot = Network_Plot,MeanMedian_DF = MeanMedian_DF)
        
}

# CancerSubtype_result_list <- lapply(names(TCGA_PanCan_EXPKeep_tmpList), CancerTypeDF_list_analysis, TCGA_PanCan_EXPKeep_tmpList = TCGA_PanCan_EXPKeep_testSetList, BRCAness_DF_ALL = BRCAness_DF_ALL,Top_gene_DF = Top_gene_DF, Output_Dir = CancerSubtype_Testsetresult_path)
# names(CancerSubtype_result_list) <- names(TCGA_PanCan_EXPKeep_tmpList)

Get_CancerType_GCNNPerformance_resultswithTTest <- function(CancerSubtype_result_list,CancerSubtype_Testsetresult_list,CancerType_GCNNPerformance_results){
  TTest_extracted_data <- lapply(CancerSubtype_result_list, function(sub_list) {
    data.frame(
      statistic_med = sub_list$TTest_df$statistic[1],
      pvalue_med = sub_list$TTest_df$p.value[1],
      statistic_mean = sub_list$TTest_df$statistic[2],
      pvalue_mean = sub_list$TTest_df$p.value[2]
    )
  })
  TTestdf_extracted <- bind_rows(TTest_extracted_data, .id = "cancer type abbreviation")
  
  Testset_TTest_extracted_data <- lapply(CancerSubtype_Testsetresult_list, function(sub_list) {
    data.frame(
      TestSet_statistic_med = sub_list$TTest_df$statistic[1],
      TestSet_pvalue_med = sub_list$TTest_df$p.value[1],
      TestSet_statistic_mean = sub_list$TTest_df$statistic[2],
      TestSet_pvalue_mean = sub_list$TTest_df$p.value[2]
    )
  })
  Testset_TTestdf_extracted <- bind_rows(Testset_TTest_extracted_data, .id = "cancer type abbreviation")
  CancerType_GCNNPerformance_resultswithTTest <- full_join(TTestdf_extracted,CancerType_GCNNPerformance_results, by = "cancer type abbreviation") %>%
    full_join(Testset_TTestdf_extracted, by = "cancer type abbreviation")
  return(as.data.frame(CancerType_GCNNPerformance_resultswithTTest))
}

result_Output <- function(OutputDir,FinalResultList){
  # FinalResultList$resampled_cancerSubtype_plot
  resampled_cancerSubtype_plot_path <- file.path(OutputDir, "resampled_cancerSubtype_plot.png")
  Testset_cancerSubtype_plot_path <- file.path(OutputDir, "Testset_cancerSubtype_plot.png")
  CancerType_GCNNPerformance_resultswithTTest_path <- file.path(OutputDir, "CancerType_GCNNPerformance_resultswithTTest.csv")
  
  ## output to 6k resolution image
  ggsave(resampled_cancerSubtype_plot_path, plot = FinalResultList$resampled_cancerSubtype_plot, width = 6144, height = 3160, dpi = 300,units = "px") 
  ggsave(Testset_cancerSubtype_plot_path, plot = FinalResultList$Testset_cancerSubtype_plot, width = 6144, height = 3160, dpi = 300,units = "px")
  write.csv(FinalResultList$CancerType_GCNNPerformance_resultswithTTest, CancerType_GCNNPerformance_resultswithTTest_path, row.names = FALSE, na="") # Set row.names = FALSE if you don't want row names saved
}


post_Exp_TopgeneHeatmapNetwork_Create<- function(TCGA_PanCan_EXP, BRCAness_Label, TOP_genes, cancer_genes, trans_reg_genes, Clinical_DF,reactome_FIs_TB,pathway_gene_list,LRP_class_file){
  ## Preprocess dataframe
  DataFrame_list <- DataFrame_Preprocess(BRCAness_DF,TCGA_PanCan_EXP,TOP_genes,Clinical_DF)
  TCGA_PanCan_EXP <- DataFrame_list$TCGA_PanCan_EXP
  BRCAness_DF <- DataFrame_list$BRCAness_DF
  Top_gene_DF <- DataFrame_list$Top_gene_DF
  CancerSample_DF <- DataFrame_list$CancerSample_DF
  
  ## Create output dir
  OutputDir <- create__HeatmapNetwork_OutputDir(LRP_class_file,"figures/Heatmap_Network_result")
  CancerSubtype_result_path <- file.path(OutputDir, "CancerSubtype_result")
  dir.create(CancerSubtype_result_path,showWarnings = F)
  CancerSubtype_Testsetresult_path <- file.path(OutputDir, "CancerSubtype_Testsetresult") 
  dir.create(CancerSubtype_Testsetresult_path,showWarnings = F)
  
  ## Get CancerType_GCNNPerformance_results dataframe and TCGA_PanCan_Testset_EXPKeep/TCGA_PanCan_Testset_EXPKeep 
  CancerType_GCNNPerformance_results_list <- Get_CancerType_GCNNPerformance_results(Clinical_DF,LRP_class_file,DataFrame_list$TCGA_PanCan_EXP,CancerSample_DF)
  CancerType_GCNNPerformance_results<- CancerType_GCNNPerformance_results_list$CancerType_GCNNPerformance_results
  Testset_cancerSubtype_plot <- CancerType_GCNNPerformance_results_list$Testset_cancerSubtype_plot
  TCGA_PanCan_Testset_EXPKeep <- CancerType_GCNNPerformance_results_list$TCGA_PanCan_Testset_EXPKeep
  TCGA_PanCan_EXPKeep <- TCGA_PanCan_EXP[TCGA_PanCan_EXP$probe %in% TOP_genes,]
  
  
  ### split to cancer types dataframe for visualization
  TCGA_PanCan_EXPKeep_tmpList <- TCGA_PanCan_EXPKeep_ListCreate(TCGA_PanCan_EXPKeep,CancerSample_DF)
  TCGA_PanCan_EXPKeep_testSetList <- TCGA_PanCan_EXPKeep_ListCreate(TCGA_PanCan_Testset_EXPKeep,CancerSample_DF)

  ## get BRCAness_DF_ALL
  BRCAness_DF_ALL <- BRCAness_DF %>% dplyr::rename(BRCAness= 1) %>% tibble::rownames_to_column( "sample") %>% mutate(BRCAlabel = ifelse(BRCAness==1,"BRCAness","NonBRCAness")) %>%
    arrange(BRCAlabel) %>% left_join(Clinical_DF) %>% dplyr::rename(`Cancer Type`= `cancer type abbreviation`)
  
  ## create the barplot for resampled/labeled samples.
  resampled_cancerSubtype_plot <- Get_resampled_Barplot(BRCAness_DF_ALL)
  
  ## Heatmap and network plot for all data (training + test set)
  CancerSubtype_result_list <- lapply(names(TCGA_PanCan_EXPKeep_tmpList), CancerTypeDF_list_analysis,TCGA_PanCan_EXPKeep_tmpList = TCGA_PanCan_EXPKeep_tmpList, BRCAness_DF_ALL = BRCAness_DF_ALL,
                                      Top_gene_DF = Top_gene_DF,Output_Dir = CancerSubtype_result_path)
  names(CancerSubtype_result_list) <- names(TCGA_PanCan_EXPKeep_tmpList)
  
  ## Heatmap and network plot for test set data 
  CancerSubtype_Testsetresult_list <- lapply(names(TCGA_PanCan_EXPKeep_testSetList), CancerTypeDF_list_analysis,TCGA_PanCan_EXPKeep_tmpList = TCGA_PanCan_EXPKeep_testSetList, BRCAness_DF_ALL = BRCAness_DF_ALL,
                                             Top_gene_DF = Top_gene_DF,Output_Dir = CancerSubtype_Testsetresult_path)
  names(CancerSubtype_Testsetresult_list) <- names(TCGA_PanCan_EXPKeep_testSetList)
  
  ## Get CancerType_GCNNPerformance_resultswithTTest Dataframe
  CancerType_GCNNPerformance_resultswithTTest <- Get_CancerType_GCNNPerformance_resultswithTTest(CancerSubtype_result_list,CancerSubtype_Testsetresult_list,CancerType_GCNNPerformance_results)
  
  FinalResultList <- list(CancerSubtype_result_list = CancerSubtype_result_list, CancerSubtype_Testsetresult_list = CancerSubtype_Testsetresult_list,CancerType_GCNNPerformance_resultswithTTest = CancerType_GCNNPerformance_resultswithTTest, 
                          resampled_cancerSubtype_plot = resampled_cancerSubtype_plot, Testset_cancerSubtype_plot = Testset_cancerSubtype_plot)
  
  ## output to directory
  result_Output(OutputDir,FinalResultList)
  
  # CancerSubtype_result_list$ALL$Network_Plot
  # CancerSubtype_result_list$BLCA$BRCAness_p
  # TCGA_PanCan_EXPKeep_tmpListDF <- TCGA_PanCan_EXPKeep_tmpList$BLCA
  return(FinalResultList)
}
FinalResultList <- post_Exp_TopgeneHeatmapNetwork_Create(TCGA_PanCan_EXP, BRCAness_Label, TOP_genes, cancer_genes, trans_reg_genes, Clinical_DF,reactome_FIs_TB,pathway_gene_list,LRP_class_file)
FinalResultList$CancerSubtype_Testsetresult_list$ALL$BRCAness_p


########################### Top 20 DE GLRP genes enrichment analysis across all cancer subtypes
library(tidyverse)
CancerSubtype_results_path <- "./figures/Heatmap_Network_result/CancerSubtype_result"
files <- list.files(path = CancerSubtype_results_path, pattern = "_MeanMedian_DF.csv")
df_list <- lapply(file.path(CancerSubtype_results_path,files), read.csv)
names(df_list) <- sapply(files, function(x) strsplit(x, "_")[[1]][1])

## Specify top_n genes for Pathway enrichment analysis across multiple cancer types
top_n <- 20
get_top_probes <- function(df,top_n) {
  # Calculate the absolute values of the Med_diff column and add it as a new column
  df$Abs_Med_diff <- abs(df$Med_diff)
  
  # Order the data frame by the absolute Med_diff values in descending order
  df_ordered <- df[order(-df$Abs_Med_diff), ]
  
  # Select the top 10 probes based on the absolute Med_diff values
  top_probes <- df_ordered$probe[1:top_n]
  
  # Return the character vector of top 10 probes
  return(top_probes)
}

# test <- get_top_probes(df_list$BRCA,20)
top_genes_list <- lapply(df_list, get_top_probes, top_n = top_n)
# Apply the find_pathways function to top_probes_list
Pathway_results_list <- lapply(names(top_genes_list), function(name) {
  top_genes <- top_genes_list[[name]]
  output_filename <- name  # This is the name of the list element, e.g., "BRCA"
  find_pathways(top_genes, CancerSubtype_results_path, output_filename)
})
names(Pathway_results_list) <- names(top_genes_list)

GO_results_list <- lapply(names(top_genes_list), function(name) {
  top_genes <- top_genes_list[[name]]
  output_filename <- name  # This is the name of the list element, e.g., "BRCA"
  find_GO_terms(top_genes, CancerSubtype_results_path, output_filename)
})
names(GO_results_list) <- names(top_genes_list)


## summarize the enriched pathway counts across multiple cancer types
PathwayResult_Output_dir <- "./figures/Heatmap_Network_result"
count_pathway_occurrences <- function(results_list,Output_dir) {
  output_filename <- "Summarized"
  # Initialize an empty data frame to store the aggregated counts and types
  pathway_counts <- data.frame(ID = character(), Description = character(), Count = integer(), Cancer_type = character(), stringsAsFactors = FALSE)
  
  # Loop through each element in the results_list
  for (name in names(results_list)) {
    # Check if the dataframe is empty
    if (nrow(results_list[[name]]) == 0) {
      next # Skip this iteration if the dataframe is empty
    }
    
    # Extract the current dataframe
    current_df <- results_list[[name]]
    
    # Create a dataframe of counts of unique IDs
    current_counts <- as.data.frame(table(current_df$ID))
    
    # Rename the columns for clarity
    colnames(current_counts) <- c("ID", "Count")
    
    # Merge with the descriptions
    current_counts <- merge(current_counts, current_df[, c("ID", "Description")], by = "ID", all.x = TRUE)
    
    # Add the Cancer_type column
    current_counts$Cancer_type <- name
    
    # Sum up the counts for each ID
    pathway_counts <- rbind(pathway_counts, current_counts)
  }
  
  # Aggregate the counts for each ID
  aggregated_counts <- aggregate(Count ~ ID + Description, data = pathway_counts, FUN = sum)
  
  # Create a list of cancer types for each ID
  cancer_types <- split(pathway_counts$Cancer_type, pathway_counts$ID)
  cancer_type_column <- sapply(cancer_types, function(x) paste(unique(x), collapse = ", "))
  
  # Add the Cancer_type column to the aggregated counts
  aggregated_counts$Cancer_type <- cancer_type_column[match(aggregated_counts$ID, names(cancer_type_column))]
  
  # Order by descending count
  aggregated_counts <- aggregated_counts[order(-aggregated_counts$Count), ]
  
  ## output 
  Pathway_result_path <- file.path(Output_dir,paste0(output_filename, "_Pathway_result.csv"))
  write.table(aggregated_counts,Pathway_result_path, sep = ",",row.names = FALSE,col.names = TRUE)
  # Return the aggregated count dataframe
  return(aggregated_counts)
}

Pathway_SumResult_PanCancers <- count_pathway_occurrences(Pathway_results_list,PathwayResult_Output_dir)


