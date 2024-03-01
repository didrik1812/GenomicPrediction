

rm(list = ls())
path_to_results <- "~/../../../../work/didrikls/GenomicPrediction/models/results.feather"
read_path  <- "~/../../../../work/didrikls/GenomicPrediction/data/interim/"
setwd("~/../../../../work/didrikls/GenomicPrediction/src")
config_yaml <- yaml::read_yaml("../config.yaml")
phenotype <- config_yaml$phenotype
model <- config_yaml$model
if (model != "GAM"){
	quit(save="no")
}

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
    install.packages("~/ProjectThesis/code/SMisc.tar.gz", repos = NULL, type = "source")
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


library(mgcv)
library(nadiv)
library(pedigree)
library(MASS)
library(MCMCpack)
library(MCMCglmm)
# This is a self-made package that I send you to install locally:
library(SMisc) # Take contact if you do not have this


library(dplyr)

lsource("h_dataPrep.r")

# Some data wranging to ensure that the IDs in the data correspond to the IDs in the A and G-matrices (nothing to worry about):
# indicates that some IDs are missing:
# d.map[3110:3125, ]
# from this we see the number of anmals
Nanimals <- 3116

# remove missing values
d.morph <- filter(d.morph, !is.na(eval(as.symbol(phenotype))))
# names(d.morph)
# In the reduced pedigree only Nanimals out of the 3147 IDs are preset.
d.map$IDC <- 1:nrow(d.map)

d.morph$IDC <- d.map[match(d.morph$ringnr, d.map$ringnr), "IDC"]
### Prepare for use in INLA -
d.morph$IDC4 <- d.morph$IDC3 <- d.morph$IDC2 <- d.morph$IDC

corr_cvs_EG <- c()
corr_cvs_G <- c()

for (i in 0:9) {
    # get CV indices
    ringnr_train <- pull(arrow::read_feather(paste(data_path, "temp/ringnr_train_", i, ".feather", sep = "")), "ringnr")
    ringnr_test <- pull(arrow::read_feather(paste(data_path, "temp/ringnr_test_", i, ".feather", sep = "")), "ringnr")


    # make test and train set
    d.morph_train <- filter(d.morph, !ringnr %in% ringnr_test)
    d.morph_test <- filter(d.morph, ringnr %in% ringnr_test)

    n_train <- dim(d.morph_train)[1]
    n_test <- dim(d.morph_test)[1]
    N <- n_train + n_test

    # Save the phenotypic value in the test set, if we only looking at genetic effects (two-step) we take the average
    pheno_test_EG <- d.morph_test[, phenotype]
    pheno_test <- as.data.frame(d.morph_test %>%
        group_by(ringnr) %>%
        summarize(
            mean_pheno = mean(eval(as.symbol(phenotype)))
        ))[, "mean_pheno"]

    # However, INLA has no predict function, so have to fill the test-values with NAs and then merge it back into the train-set
    # d.morph_test[, phenotype] <- NA
    # d.morph_train <- union_all(d.morph_train, d.morph_test)

    # names(d.morph_train)
    # All individuals
    idxs <- 1:Nanimals
    # get the indicies corresponding to the individuals in the test set
    idxs_test <- which(d.map$ringnr %in% unique(ringnr_test))
    # TODO: Model studff goes here ->



 corr_EG <- cor(preds_EG, pheno_test_EG, method = "pearson")
    corr <- cor(preds, pheno_test, method = "pearson")
    cat("result of fold", i, "corr_G:", corr, "corr_EG", corr_EG, "\n")
    corr_cvs_G <- c(corr_cvs_G, corr)
    corr_cvs_EG <- c(corr_cvs_EG, corr_EG)
}
# save results
result_df <- arrow::read_feather(path_to_results)
INLA_result_df <- data.frame(name = "INLA_EG", corr = corr_cvs_EG, phenotype = phenotype)
INLA_result_df <- rbind(INLA_result_df, data.frame(name = "INLA_G", corr = corr_cvs_G,phenotype = phenotype))
result_df <- rbind(result_df, INLA_result_df)
arrow::write_feather(result_df, path_to_results)

