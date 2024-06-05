# 
# GWAS.R 
# Plot to make manhattan plots from GWAS results
# Also compares GWAS to SHAP
# Figs are saved under the reports/figures/ folder
# Script is run from prepGWAS.sh
# 

library(dplyr)
library(stringr)
library(ggplot2)
library(qqman)
library(readr)
library(data.table)
library(tidyr)
library(smplot2)
library(latex2exp)

setwd("/work/didrikls/GenomicPrediction")
args <- commandArgs(trailingOnly = TRUE)
PHENOTYPE <- args[1]
shap_path <- args[2]


# PLOT GWAS RESULTS

resultGemma <- read_table("./output/GWASresults.lmm.assoc.txt")
resultGemma <- resultGemma[!(resultGemma$chr %in% c(0, 16,30,32)),]
#compute the Bonferroni threshold
bonferroni<- -log10(0.05/ nrow(resultGemma))

pdf(paste("reports/figures/GWAS_",PHENOTYPE,".pdf",sep=""))
# png(paste("GWAS_",PHENOTYPE,".png",sep=""))
manhattan(resultGemma,chr="chr",bp="ps",p="p_lrt",snp="rs", genomewideline = bonferroni, ylim = c(0, 6),cex.axis = 1.2, cex.lab=1.5, cex.main=1.5, ylab="")
title(ylab = TeX(r"($-log_{10}(p)$)"),line=2, cex.lab=1.5)
dev.off()

# COMPARE GWAS TO SHAP

shap <- arrow::read_feather(shap_path)

shap <- t(shap)
shap <- data.frame(snp = rownames(shap), shap = shap[,1], row.names = NULL)


get_bp <- function(snp) {
    # snp <- str_split(snp, "[ai]")[[1]][2]
    snp <- str_split(snp, "[_]")[[1]][1]
    return(snp[[1]])
}

shap$rs <- lapply(shap$snp, get_bp)

shap <- as.data.frame(lapply(shap, unlist))
head(shap)


eps <-1e-4
shap_gwas <- merge(shap, resultGemma,by = "rs")
shap_gwas$log_p = -log10(shap_gwas$p_lrt)
shap_gwas$shap = as.numeric(shap_gwas$shap)
shap_gwas$log_shap = log10(shap_gwas$shap + eps)

pdf(paste("reports/figures/GWAS_SHAP_",PHENOTYPE,".pdf",sep=""))
ggplot(shap_gwas, aes(x = log_p, y = shap))+geom_point()+theme_bw(base_size = 22)+sm_statCorr()
dev.off()

cor(shap_gwas$log_p, shap_gwas$shap)
# REMOVE SNPs with SMALL SHAP VALUES
shap_gwas_non_zero <- shap_gwas[shap_gwas$shap >eps,]
pdf(paste("reports/figures/NONZERO_GWAS_SHAP_",PHENOTYPE,".pdf",sep=""))
ggplot(shap_gwas_non_zero, aes(x = log_p, y = shap))+geom_point()+theme_bw(base_size = 22)+sm_statCorr()
dev.off()
length(shap_gwas_non_zero$shap)
cor(shap_gwas_non_zero$log_p, shap_gwas_non_zero$shap)

# arrow::write_feather(shap_gwas, "shap_gwas_body_mass.feather")


# df = arrow::read_feather("shap_gwas_tarsus.feather")

# df_sub = df[,c("snp","chr" ,"shap", "p_lrt")]

# df_sub[order(df_sub$shap, decreasing = T),]
# dim(df_sub[df_sub$shap > 0.0,])
# dim(df_sub[df_sub$shap > 1e-4,])

