# install.packages("qqman")
library(qqman)
setwd("/work/didrikls/GenomicPrediction")
# shap_path <- "models/xgboostMass_70kBV/shap.feather"
shap_path <- "models/linearlgbmTarsus_70kBV/coefs.feather"
shap_vals <- arrow::read_feather(shap_path)
head(shap_vals)[1:10]



library(tidyr)
library(dplyr)
library(stringr)
map_path <- "data/raw/combined_200k_70k_helgeland_south_corrected_snpfiltered_2024-02-05.map"
# map_path <- "data/raw/combined_200k_70k_helgeland_south_corrected_snpfiltered_2024-02-05.map"
chro <- read.table(map_path, header = F)
names(chro) <- c("chr", "snpID", "value", "bp")

snp_to_discard <- chro[chro$chr %in% c(0, 30, 32), "snpID"]
length(unique(snp_to_discard))

SNP_cols <- names(shap_vals)
tmpdf <- data.frame(origsnp = SNP_cols, snpID = NA)

get_bp <- function(snp) {
    # snp <- str_split(snp, "[ai]")[[1]][2]
    snp <- str_split(snp, "[_]")[[1]][1]
    return(snp[[1]])
}

tmpdf$snpID <- lapply(tmpdf$origsnp, get_bp)

tmpdf2 <- as.data.frame(lapply(tmpdf, unlist))

tmpdfchro <- merge(tmpdf2, chro[, c("chr", "snpID", "bp")], by = "snpID")
snp_to_keep <- tmpdf[!(tmpdf$snpID %in% snp_to_discard), "origsnp"]


sum(chro$SNP %in% tmpdf$snpID)

data_red <- shap_vals[, c(snp_to_keep)]
mean_shap <- colMeans(data_red)

sub_df <- head(mean_shap)
# rshap <- data.frame(origsnp = names(mean_shap), shap = mean_shap)
rshap <- data.frame(origsnp = names(mean_shap), coeff = mean_shap)


library(grid)
library(gridGraphics)
# install.packages("gridGraphics")
library(ggplot2)
manhattan_df <- merge(rshap, tmpdfchro, by = "origsnp")
head(manhattan_df)
# manhattan(manhattan_df, chr = "chr", bp = "bp", p = "shap", snp = "snpID", logp = FALSE, ylim = c(0, max(manhattan_df$shap)), ylab = "mean SHAP")
# g <- grid.grabExpr(grid.echo(p))
# manhattan(manhattan_df, chr = "chr", bp = "bp", p = "shap", snp = "snpID", logp = FALSE, ylim = c(0, 0.015), ylab = "mean SHAP")
manhattan(manhattan_df, chr = "chr", bp = "bp", p = "coeff", snp = "snpID", logp = FALSE, ylim = c(0, max(manhattan_df$coeff)), ylab = "mean coeff")
