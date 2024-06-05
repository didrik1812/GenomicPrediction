# manhatten.R
# Plot SHAP values in a manhatten plot
# Run by the prepGWAS.sh script



# install.packages("qqman")
library(qqman)
library(tidyr)
library(dplyr)
library(stringr)

setwd("/work/didrikls/GenomicPrediction")
# shap path is extracted from arguments
args <- commandArgs(trailingOnly = TRUE)
shap_path <- args[1]
save_name_list <- str_split(shap_path, "/")[[1]]
save_name <- save_name_list[length(save_name_list) - 1]
shap_vals <- arrow::read_feather(shap_path)






# Some prepping is needed to match the SHAP values with chromosome loccations
# NED A MAP FILE, gives where each SNP is loccated
map_path <- "data/raw/combined_200k_70k_helgeland_south_corrected_snpfiltered_2024-02-05.map"
# map_path <- "data/raw/combined_200k_70k_helgeland_south_corrected_snpfiltered_2024-02-05.map"
chro <- read.table(map_path, header = F)

names(chro) <- c("chr", "snpID", "value", "bp")

snp_to_discard <- chro[chro$chr %in% c(0, 30, 32), "snpID"]

SNP_cols <- names(shap_vals)
tmpdf <- data.frame(origsnp = SNP_cols, snpID = NA)
# Need to split the SNP name (in the shap df) to get the basepair
get_bp <- function(snp) {
    # snp <- str_split(snp, "[ai]")[[1]][2]
    snp <- str_split(snp, "[_]")[[1]][1]
    return(snp[[1]])
}

tmpdf$snpID <- lapply(tmpdf$origsnp, get_bp)
# make df instead of list
tmpdf2 <- as.data.frame(lapply(tmpdf, unlist))

# merge the map-file stuff with the SNPname from the SNP columns in shap
tmpdfchro <- merge(tmpdf2, chro[, c("chr", "snpID", "bp")], by = "snpID")
snp_to_keep <- tmpdf[!(tmpdf$snpID %in% snp_to_discard), "origsnp"]
# we now have a mapping between SHAP SNPs and the chromosome loccation

sum(chro$SNP %in% tmpdf$snpID)


# remove some SNP on chromosomes not desired
mean_shap <- shap_vals[, c(snp_to_keep)]

rshap <- t(mean_shap)
rshap_2 <- data.frame(origsnp = rownames(rshap), shap = rshap[, 1], row.names = NULL)

library(grid)
# install.packages("gridGraphics")
library(gridGraphics)
library(ggplot2)
library(dplyr)
# merge shap with loccation
manhattan_df <- merge(rshap_2, tmpdfchro, by = "origsnp")
# NOW WE CAN PLOT
pdf(paste("reports/figures/SHAP", save_name, ".pdf", sep = ""))
par(mar = c(5, 6, 4, 1) + .1)
manhattan(manhattan_df, chr = "chr", bp = "bp", p = "shap", snp = "snpID", logp = FALSE, ylim = c(0, 0.010), , cex.axis = 1.2, cex.lab = 1.5, cex.main = 1.5, ylab = "")
title(ylab = "mean |SHAP|", line = 4, cex.lab = 1.5)
dev.off()
