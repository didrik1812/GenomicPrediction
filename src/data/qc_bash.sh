#!/bin/bash

#######################################################
# qc_bash.sh
# script used to make the SNP matrix for the 70K data
# Uses plink
# run it by "sh qc_bash.sh"
########################################################


# Set variables
fam_file="data/raw/combined_200k_70k_helgeland_south_corrected_snpfiltered_2024-02-05.fam" # FILL IN PATH TO FAM FILE
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
plink --bfile "$file_root" \
              --recode A \
              --maf "$maf" \
              --geno "$genorate_snp" \
              --mind "$genorate_ind" \
              --chr-set 32 \
              --memory "$mem" \
              --keep keep.txt \
              --threads "$ncores" \
              --out qc

# Check exit code
exit_code=$?
if [ $exit_code -ne 0 ]; then
  echo "Error in plink"
  exit $exit_code
fi

exit $exit_code

