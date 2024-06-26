# A small helper script to prep data before GWAS

library(dplyr)
library(stringr)
library(ggplot2)
library(qqman)
library(readr)
setwd("/work/didrikls/GenomicPrediction/data")
args <- commandArgs(trailingOnly = TRUE)
PHENOTYPE <- args[1]

NAME <- "combined_200k_70k_helgeland_south_corrected_snpfiltered_2024-02-05"

morphData <- read.table("raw/AdultMorphology_20240201_fix.csv", header = T, sep = ";") # sep="\t")

tmp <- morphData %>%
    mutate(FID = ringnr) %>%
    rename(IID = ringnr) %>%
    select(FID, IID, PHENOTYPE) %>%
    write_delim(paste("GWAS_", PHENOTYPE, ".txt", sep = ""), delim = " ")
