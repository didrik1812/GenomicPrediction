library(data.table)
Sys.setenv(PATH=paste(Sys.getenv("PATH"), "~/plink", sep=":"))
do_qc <- function(fam_file, ## PATH TO .FAM file
                  ncores, # how many cores plink gets to use
                  mem, # how much memory plink gets to use
                  qc_filt, # list of quality control filters
                  plink_path # path to PLINK program
                  ) {
  
  file_root <- gsub(pattern = ".fam", replacement = "", x = fam_file)
  fam <- fread(fam_file, select = c(1, 2), data.table = FALSE, header = FALSE)
  # Exclude samples with high heterozygosity
  fam_keep <- fam[!grepl(pattern = ".*HIGHHET.*", x = fam$V2), ]
  # Exclude samples with mismatches in sex
  fam_keep <- fam_keep[!grepl(pattern = ".*MISSEX.*", x = fam_keep$V2), ]
  
  # For inds. genotyped multiple times, keep last one
  fam_keep <- fam_keep[!duplicated(fam_keep$V1, fromLast = TRUE), ]
  
  write.table(fam_keep,
              file = paste0("keep.txt"),
              quote = FALSE,
              row.names = FALSE,
              col.names = FALSE)
  
  exit_code <- # system2() calls the command line
    system2(plink_path,
            paste0("--bfile ", file_root, " ",
                   "--recode A ",
                   "--maf ", qc_filt$maf, " ", # Filter by minor allele frequency
                   "--geno ", qc_filt$genorate_snp, " ", # Filter SNPs by call rate
                   "--mind ", qc_filt$genorate_ind, " ", # Filter inds by call rate
                   "--chr-set 32 ", # Sparrow chromosomes
                   "--memory ", mem, " ",
                   "--keep keep.txt ",
                   "--threads ", ncores, " ",
                   "--out qc")) # Name of output file, feel free to change
  
  if (exit_code != 0) {
    stop("Error in plink")
  }
  
  exit_code
}

qc_filt <- list(genorate_ind = 0.05,
                genorate_snp = 0.1,
                maf = 0.01)

do_qc(qc_filt = qc_filt,
      mem = 8000,
      ncores = 6,
      fam_file = "data/raw/combined_200k_70k_helgeland_south_corrected_snpfiltered_2024-02-05.fam", # FILL IN PATH TO FAM FILE,
      plink_path = "plink" # FILL IN PATH TO PLINK PROGRAM
        )
