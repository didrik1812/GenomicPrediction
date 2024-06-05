#!/bin/bash

# This script preps stuff for GWAS and then runs GEMMA, plots the results and compare it to SHAP values
# run by sh prepGWAS.sh 
# Modify these variables to get desired results

phenotype="thr_tarsus" # name of phenotype in morph data (thr_tarsus or body_mass)
phenotype_plot="tarsus" # name of phenotype in plot
shap_path="models/xgboost2TarsusStd_70kExtBV/shap2.feather" # path to the shap feather file to use for comparison


# prep the phenotype first
nice Rscript /work/didrikls/GenomicPrediction/src/visualization/prepGWAS.R "$phenotype"
# after prepping the data is stored here 
path_to_prepped_data="/work/didrikls/GenomicPrediction/data/GWAS_${phenotype}.txt" 

# Make Manhatten for SHAP
nice Rscript /work/didrikls/GenomicPrediction/src/visualization/manhatten.R "$shap_path"

# Set variables
fam_file="/work/didrikls/GenomicPrediction/data/raw/combined_200k_70k_helgeland_south_corrected_snpfiltered_2024-02-05.fam" # FILL IN PATH TO FAM FILE
mem=8000
ncores=6
genorate_ind=0.05
genorate_snp=0.1
maf=0.01


# Extract file root
file_root="${fam_file%.fam}"

# Exclude samples with high heterozygosity
grep -v ".*HIGHHET.*" "$fam_file" > fam_keep.tmp

# Exclude samples with mismatches in sex
grep -v ".*MISSEX.*" fam_keep.tmp > fam_keep.tmp2

# For inds. genotyped multiple times, keep last one
awk '!seen[$1]++' fam_keep.tmp2 > fam_keep.tmp3

# Write fam_keep to keep.txt
awk '{print $1,$2}' fam_keep.tmp3 > keep.txt

# Run PLINK command
$HOME/plink/./plink --bfile "$file_root" \
              --recode A \
              --maf "$maf" \
              --geno "$genorate_snp" \
              --mind "$genorate_ind" \
              --chr-set 32 \
              --nonfounders \
              --pheno "$path_to_prepped_data"\
              --pheno-name "$phenotype" \
              --make-bed \
              --memory "$mem" \
              --keep keep.txt \
              --threads "$ncores" \
              --out inputForGemma


#########################
# Run GWAS with GEMMA
#########################
# compute the relationship matrix for population structure correction

$HOME/./gemma-0.98.5-linux-static-AMD64 -bfile inputForGemma -gk 1 -o RelMat

$HOME/./gemma-0.98.5-linux-static-AMD64 -bfile inputForGemma -k ./output/RelMat.cXX.txt -lmm 2 -o GWASresults.lmm

# Plot GWAS
nice Rscript /work/didrikls/GenomicPrediction/src/visualization/GWAS.R "$phenotype_plot" "$shap_path"

# Check exit code
exit_code=$?
if [ $exit_code -ne 0 ]; then
  echo "Error in plink or gemma"
  exit $exit_code
fi

exit $exit_code

