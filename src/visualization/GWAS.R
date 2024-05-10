library(dplyr)
library(stringr)
library(ggplot2)
library(qqman)
library(readr)
library(data.table)
library(tidyr)
library(smplot2)
setwd("/work/didrikls/GenomicPrediction")
PHENOTYPE = "tarsus"



resultGemma <- read_table("./output/GWASresults.lmm.assoc.txt")
#compute the Bonferroni threshold
bonferroni<- -log10(0.05/ nrow(resultGemma))

pdf(paste("reports/figures/GWAS_",PHENOTYPE,".pdf",sep=""))
#png(paste("GWAS_",PHENOTYPE,".png",sep=""))
manhattan(resultGemma,chr="chr",bp="ps",p="p_lrt",snp="rs",genomewideline=bonferroni)
dev.off()


shap_path <- "models/xgboostTarsus_70kBV/shap.feather"

shap <- arrow::read_feather(shap_path)
# shap <- shap %>% colMeans()
shap <- data.frame(cbind(names(shap), colMeans(shap)))
colnames(shap) <- c("snp", "shap")
rownames(shap) <- NULL

get_bp <- function(snp) {
    # snp <- str_split(snp, "[ai]")[[1]][2]
    snp <- str_split(snp, "[_]")[[1]][1]
    return(snp[[1]])
}

shap$rs <- lapply(shap$snp, get_bp)

shap <- as.data.frame(lapply(shap, unlist))
head(shap)


# shap <- shap %>% pivot_longer(cols = everything(), names_to = "rs", values_to = "shap")

shap_gwas <- merge(shap, resultGemma,by = "rs" )
shap_gwas$log_p = -log10(shap_gwas$p_lrt)
shap_gwas$shap = as.numeric(shap_gwas$shap)
shap_gwas$log_shap = log10(shap_gwas$shap + eps)

pdf(paste("reports/figures/GWAS_SHAP_",PHENOTYPE,".pdf",sep=""))
ggplot(shap_gwas, aes(x = log_p, y = shap))+geom_point()+theme_bw()+sm_statCorr()
dev.off()

cor(shap_gwas$log_p, shap_gwas$shap)

eps <-1e-4
shap_gwas_non_zero <- shap_gwas[shap_gwas$shap >eps,]
pdf(paste("reports/figures/NONZERO_GWAS_SHAP_",PHENOTYPE,".pdf",sep=""))
ggplot(shap_gwas_non_zero, aes(x = log_p, y = shap))+geom_point()+theme_bw()+sm_statCorr()
dev.off()
length(shap_gwas_non_zero$shap)
cor(shap_gwas_non_zero$log_p, shap_gwas_non_zero$shap)
