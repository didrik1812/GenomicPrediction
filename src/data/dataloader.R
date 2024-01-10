#######################################################################
## Didrik Sand, September 2024
# Script is based on code from Stefanie Muff
## This script is used to extract the data needed for the two-step approach
# The script should not be run directly, but rather from the configuration script code/model_exploration.py
#######################################################################

# CHANGE THIS TO YOUR OWN PATH: (i.e where the data is stored)
store_path <- "~/../../../../work/didrikls/GenomicPrediction/data/processed/"
setwd("~/../../../../work/didrikls/GenomicPrediction/src")
args <- commandArgs(trailingOnly = TRUE)
phenotype <- args[1]

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
d.map[3110:3125, ]
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

# RUN LMM ON PHENOTYPE TO SEPARETE ID EFFECT FROM ENVIRONMENTAL EFFECTS
d.mean.pheno <- as.data.frame(d.pheno %>%
    group_by(ringnr) %>%
    summarize(mean_pheno = mean(eval(as.symbol(phenotype)))))

formula.pheno.lmm <- eval(as.symbol(phenotype)) ~   sex + FGRM + month + age + outer + other +
    (1 | island_current) +
    (1 | hatchyear) +
    (1 | ringnr)


library(lme4)
dd <- d.morph[!is.na(d.morph[phenotype]), ]
r.pheno.lmer <- lmer(formula.pheno.lmm, data = dd)

# Residuals
d.pheno.res <- data.frame(ringnr = dd$ringnr, pheno_res = residuals(r.pheno.lmer))

# Mean over repeats for each individual
d.mean.pheno.res <- as.data.frame(d.pheno.res %>%
    group_by(ringnr) %>%
    summarize(mean_pheno_res = mean(pheno_res)))

# ID effect
d.ID.pheno <- data.frame(ringnr = d.mean.pheno[, 1], ID.mass = ranef(r.pheno.lmer)$ringnr)


# We take as the new phenotype the estimated ID effect:
d.ID.pheno <- data.frame(ringnr = d.mean.pheno[, 1], ID = d.ID.pheno[, 2], mean_pheno = d.mean.pheno$mean_pheno)

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
sum(unique(d.pheno$ringnr) %in% unique(d.map$ringnr))
sum(d.ID.pheno$ringnr %in% SNP.matrix$ringnr)
length(unique(SNP.matrix$ringnr))

SNP.matrix.reduced <- cbind(
    SNP.matrix[, 1:6],
    (SNP.matrix[, sort(sample(7:181369, no_snps, replace = FALSE))])
)


# Generate a data frame where individuals with ring numbers from d.ID.res.mass are contained, as well as the phenotype (here the residuals from the lmer analysis with mass as response)
d.dat <- merge(d.ID.pheno[, c("ringnr", "ID")], SNP.matrix.reduced, by = "ringnr")
d.dat.full <- merge(d.ID.pheno[, c("ringnr", "ID", "mean_pheno")], SNP.matrix, by = "ringnr")
head(colnames(d.dat.full), 10)
# SAVE THE FULL DATA SET:
write_feather(d.dat.full, paste(store_path, phenotype, ".feather", sep = ""))
