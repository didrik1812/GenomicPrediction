#######################################################################
## Didrik Sand, September 2024
# Script is based on code from Stefanie Muff
## This script is used to extract the data needed for the one-step approach
# The script should not be run directly, but rather from the configuration script code/model_exploration.py
#######################################################################


# setwd("C:\\Users\\didri\\OneDrive - NTNU\\9.semester\\Prosjekt\\ProjectThesis\\code")

# CHANGE THIS TO YOUR OWN PATH: (i.e where the data is stored)
store_path <- "~/../../../../work/didrikls/GenomicPrediction/data/processed/"
setwd("~/../../../../work/didrikls/GenomicPrediction/src")
args <- commandArgs(trailingOnly = TRUE)
phenotype <- args[1]
print(getwd())
# Packages needed for the script to run:
if (!require(nadiv)) {
  install.packages("nadiv", repos = "http://cran.us.r-project.org", dependencies = TRUE)
}
if (!require(pedigree)) {
  install.packages("pedigree", repos = "http://cran.us.r-project.org", dependencies = TRUE)
}

if (!require(MASS)) {
  install.packages("MASS", repos = "http://cran.us.r-project.org", dependencies = TRUE)
}
if (!require(MCMCpack)) {
  install.packages("MCMCpack", repos = "http://cran.us.r-project.org", dependencies = TRUE)
}
if (!require(data.table)) {
  install.packages("data.table", repos = "http://cran.us.r-project.org", dependencies = TRUE)
}

if (!require(SMisc)) {
  install.packages("~\\ProjectThesis\\code\\SMisc.tar.gz", repos = NULL, type = "source")
}

if (!require(dplyr)) {
  install.packages("dplyr", repos = "http://cran.us.r-project.org", dependencies = TRUE)
}

if (!require(lme4)) {
  install.packages("lme4", repos = "http://cran.us.r-project.org", dependencies = TRUE)
}


if (!require(MCMCglmm)) {
  install.packages("MCMCglmm", repos = "http://cran.us.r-project.org", dependencies = TRUE)
}

if (!require(feather)) {
  install.packages("feather", repos = "http://cran.us.r-project.org", dependencies = TRUE)
}


# Sys.setenv(NOT_CRAN = "true")
# install.packages("arrow")

library(nadiv)
library(pedigree)
library(MASS)
library(MCMCpack)
library(MCMCglmm)
# This is a self-made package that I send you to install locally:
library(SMisc)

library(feather)

library(dplyr)

# library(keras)
# library(tensorflow)

## Old packages no longer needed (but kept as comments, in case we need them in the future...)
# library(MasterBayes)
# library(pedigreemm)
# library(bdsmatrix)
# library(irlba)
# library(RSpectra)
# library(dplyr)

# Data preparation helper script:
source("data/h_dataPrep.r")

# Some data wranging to ensure that the IDs in the data correspond to the IDs in the A and G-matrices (nothing to worry about):
# indicates that some IDs are missing:
d.map[3110:3125,]
# from this we see the number of anmals
Nanimals <- 3116


# In the reduced pedigree only Nanimals out of the 3147 IDs are preset.

d.map$IDC <- 1:nrow(d.map)

d.morph$IDC <- d.map[match(d.morph$ringnr, d.map$ringnr), "IDC"]

### Prepare for use in INLA -
d.morph$IDC4 <- d.morph$IDC3 <- d.morph$IDC2 <- d.morph$IDC


##############################################
### Preparations: Start with body mass and extract the id-value for each individual
##############################################

# keep all mass records that are not NA
d.pheno <- d.morph[!is.na(d.morph[phenotype]), c("ringnr", phenotype)]
names(d.pheno) <- c("ringnr", phenotype)


d.mean.pheno <- as.data.frame(d.pheno %>%
                                group_by(ringnr) %>%
                                summarize(mean_pheno = mean(eval(as.symbol(phenotype)))))


dd <- d.morph[!is.na(d.morph[phenotype]),]
names_list1 <- c("ringnr", "sex", "FGRM", "outer", "hatchyear")

names(d.mean.pheno) = c("ringnr", phenotype)

dd_red <- merge(d.mean.pheno, dd[, names_list1], by = "ringnr")

# ## This was the OLD way - I don't think it should be done:
# # We take as the new phenotype the sum of the ID effect and the mean of the residual for each individual:
# d.ID.res.mass <- data.frame(ringnr=d.mean.mass[,1],sum.ID.res = d.ID.mass[,2]+d.mean.mass.res[,2])

#############################################################
### Now we also load the raw SNP data matrix
#############################################################
library(data.table)
no_snps <- 20000

# Using the quality-controlled SNP matrix from Kenneth:
SNP.matrix <- data.frame(fread(paste(data_path, "Helgeland_01_2018_QC.raw", sep = "")))
# SNP.matrix <- data.frame(fread("data/full_imputed_dosage.raw"))

names(SNP.matrix)[2] <- "ringnr"
dim(SNP.matrix)
set.seed(323422)


# Generate a data frame where individuals with ring numbers from d.ID.res.mass are contained, as well as the phenotype (here the residuals from the lmer analysis with mass as response)
d.dat.full <- merge(dd, SNP.matrix, by = "ringnr")
# columns should be included in the data frame:
names_list2 <- c(
  phenotype, "ringnr", "age", "month", "other", "island_current", "sex", "FGRM", "outer", "hatchyear", "hatchisland" ,names(SNP.matrix)[7:length(names(SNP.matrix))]
)

# write data to feather file:
d.dat.full <- d.dat.full[, names_list2]
d.dat.full <- as.data.frame(
  d.dat.full %>% distinct()
)
head(colnames(d.dat.full), 15)
write_feather(d.dat.full, paste(store_path, phenotype, "EG", ".feather", sep = ""))
