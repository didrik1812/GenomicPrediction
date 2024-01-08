#-----------------------------------------------------------------------
# load some data 
data_path = "~/../../../../work/didrikls/ProjectThesis/data/"
d.year <- read.table(paste(data_path,"spuRvebasen-hatchyear-island_1993-2016_february2017_forSteffi.csv", sep = ""), header = T, sep=",")

# morphological traits for individuals on the 8 islands of interest:
d.morph <- read.table(paste(data_path,"RepeatedAdMorph-200K-Helgeland_20170623_extrainfo.csv", sep = ""), header=T, sep=",")# sep="\t")
# the following ringumbers must be removed, because they were duplicated:
d.remove <- read.table(paste(data_path,"Duplicates_to_remove_200kSNP.txt", sep = ""), header=F)
d.morph <- d.morph[!(d.morph$ringnr %in% d.remove[,1]),]
# strore in island variable the adult SNP island (as it was in the morph file before Dec 15 2017)
d.morph$island <- d.morph$adultSNPisland

# Some pre-modifications on data-------------------------------------------------------
#d.morph <- d.morph[d.morph$wing>0,]
d.morph[d.morph$mass <0,"mass"] <- NA
d.morph[d.morph$wing <0,"wing"] <- NA

### Import Dilans' natal island data (17.1.2018)
### Again new data on 4.6.2018
# d.natal <- read.table("../data_dispersal/TrueNatalIslands_20180604.txt", header=T, sep=" ")  #
### Again new data on 21.11.2018
#d.natal <- read.table("../data_dispersal/Dispersaldata_20181121.txt", header=T, sep=" ") 
 d.natal <- read.table(paste(data_path,"Dispersaldata_20210311.txt", sep = ""), header=T, sep=" ")
 
 d.natal$NatalIsland <- d.natal$natal.island

d.natal$islandIO <- ifelse(d.natal$NatalIsland %in% c(22,23,24,88),"outer",ifelse(d.natal$NatalIsland %in% c(20,26,27,28,38),"inner","other"))

d.natal$adultislandIO <- ifelse(d.natal$adult.island %in% c(22,23,24,88),"outer",ifelse(d.natal$adult.island %in% c(20,26,27,28,38),"inner","other"))

# Dispersal rate  between island systems:
sum(d.natal$adultislandIO!=d.natal$islandIO)/nrow(d.natal)

# Dispersal rate  between islands
sum(d.natal$disp==1)/nrow(d.natal)

 
### assign best knowledge of natal island
# the islandIO in d.year should be the genetic natal island from Dilan's data (if available), and otherwise the "firstisland" from d.year
# create therefore a new column "natalisland" that is initiated with "firstisland"
d.year$natalisland <- d.year$firstiland
d.year$natalisland <- ifelse(d.year$ringnr %in% d.natal$ringnr, d.natal[match(d.year$ringnr,d.natal$ringnr),"NatalIsland"], d.year$firstisland)

d.year$islandIO <- ifelse(d.year$natalisland %in% c(22,23,24,88),"outer",ifelse(d.year$natalisland %in% c(20,26,27,28,38),"inner","other"))



## Load the morph data from Alina, where inbreeding coefficients should be ok (as per 7.11.17)
data_morph_Alina <- read.table(paste(data_path,"steffie_inbreeding_morph_data_071117.txt", sep = ""), header = T, stringsAsFactors = F)

d.morph$FGRM <- data_morph_Alina[match(d.morph$ringnr,data_morph_Alina$IID),"FGRM"]

# a map of Ringnr and ID for the d.morph file
d.MAP <-  read.table(paste(data_path,"Ringnr_ID_Link_N3147_SNP183384.csv", sep = ""), header = T, sep=",")

#Import the pedigree; new pedigree from Alina (5.12.2017) (to be updated again) ------------------
d.ped <- read.table(paste(data_path,"SNP_pedigree_Helgeland_05122017.txt",sep = ""), header = T, sep=" ")

# Change the order for sire and dam to be compatible with the code below
d.ped <- d.ped[,c("id","sire","dam")]
#d.ped <- read.table("../data/SNP_pedigree_Helgeland_07112017.txt", header = T, sep=" ")
names(d.ped) <- c("ringnr","father","mother")

#There are again some ringnumbers from the pedigree missing in d.morph, so reduce d.morph a little:
d.morph <- d.morph[d.morph$ringnr %in% d.ped$ringnr,]

### ---------------------------------------------------
# Organizing the pedigree -- estimate and store relatedness:
# # need an ordered Pedigree for the inverseA() function:
d.ped <- nadiv::prepPed(d.ped)
#d.ped <- orderPed(d.ped)

# also replace the ringnumber by an id (1:nanimals)
d.ped$id <- 1:(nrow(d.ped))

d.map <- d.ped[,c("ringnr","id")]
d.map$ID <- d.MAP[match(d.map$ringnr,d.MAP$ringnr),"ID"]
d.map <- d.map[order(d.map$ID),]

# give mother and father the id
d.ped$mother.id <- d.map[match(d.ped$mother, d.map$ringnr),"id"]
d.ped$father.id <- d.map[match(d.ped$father, d.map$ringnr),"id"]
# and again the same to keep track:
d.ped$mother.id.original <- d.map[match(d.ped$mother, d.map$ringnr),"id"]
d.ped$father.id.original <- d.map[match(d.ped$father, d.map$ringnr),"id"]



### ---------------------------------------------------
# use full pedigree for inversion and calculation of inbreeding coefficients
A <- nadiv::makeA(d.ped[,c("id","mother.id","father.id")])
# pedigree::makeA(d.ped[,c("id","mother.id","father.id")],which=rep(TRUE,3556))
# A <- read.table("A.txt") 
Ainv <- MCMCglmm::inverseA(d.ped[,c("id","mother.id","father.id")])

### Some more modifications to the data file ----------------------------------------------------
### Add id to the morph and year data (used for the Ainv and A matrices from pedigree)
d.morph$id <- d.map[match(d.morph$ringnr, d.map$ringnr), "id"]
d.year$id <- d.map[match(d.year$ringnr, d.map$ringnr), "id"]
#d.morph$lastisland <- d.year[match(d.morph$ringnr,d.year$ringnr),"lastisland"]

### Some more data wranglig 
d.morph$island_current <- as.factor(d.morph$island_current)
d.morph$hatchyear <- as.factor(d.morph$hatchyear)
d.morph$year <- as.factor(d.morph$year)


### encode sex as 0/1 (not 1/2)
d.morph$sex <- ifelse(d.morph$sex==2,1,0)
d.morph$age <- scale(d.morph$age,scale=F)
d.morph$month <- scale(d.morph$month,scale=F)
d.morph$FGRM <- scale(d.morph$FGRM,scale=F)

d.morph <- d.morph[,c("ID","ringnr","indnr","sex","island_current","adultSNPisland","hatchisland","hatchyear","year","month","age","tarsus","wing","mass","island","FGRM","id")]

d.morph$tarsus[d.morph$tarsus==-99] <- NA

###--------------------------------------------------------------
### Genetic groups 
# Start by introducing a group variable
d.ped$GG <- rep(NA, nrow(d.ped))

# set genetic group to island where animal lived at first year that it was observed
# this corresponds to hatchisland in the d.morph file (but there are less individuals than in the pedigree)
# if animal is a dummy animal (if name length 5), assign it the island of its mother or father
 
d.ped$GG <- ifelse(# if ringnumber is a real animal with 7 digit ringnr
  nchar(as.character(d.ped$ringnr))==7,
  d.year[match(d.ped$ringnr,d.year$ringnr),"islandIO"], 
  ifelse( # otherwise if mother is a real animal
    nchar(as.character(d.ped$father))==7,
    d.year[match(d.ped$father,d.year$ringnr),"islandIO"],
    ifelse( # else if father is a real animal, assign it respective island group
      nchar(as.character(d.ped$mother))==7,
      d.year[match(d.ped$mother,d.year$ringnr),"islandIO"],
      NA
    )
  )
)



# finally, there are 271 dummy animals left without parents. Assign them genetic group of their offspring:
for (ii in 1:nrow(d.ped)){
  if (is.na(d.ped$GG[ii])){
    d.ped$GG[ii] <- d.ped[match(d.ped$ringnr[ii],d.ped$father),"GG"]
  }
  if (is.na(d.ped$GG[ii])){
    d.ped$GG[ii] <- d.ped[match(d.ped$ringnr[ii],d.ped$mother),"GG"] 
  }
}

# Repeat this one more time to fill the gaps:
for (ii in 1:nrow(d.ped)){
  if (is.na(d.ped$GG[ii])){
    d.ped$GG[ii] <- d.ped[match(d.ped$ringnr[ii],d.ped$father),"GG"]
  }
  if (is.na(d.ped$GG[ii])){
    d.ped$GG[ii] <- d.ped[match(d.ped$ringnr[ii],d.ped$mother),"GG"] 
  }
}


# if mother is missing, replace it by GG, if father is missing, replace it by GG
d.ped[is.na(d.ped$mother.id), "mother.id"] <- d.ped[is.na(d.ped$mother.id), "GG"]
d.ped[is.na(d.ped$father.id), "father.id"] <- d.ped[is.na(d.ped$father.id), "GG"]


# need q_{ij} values for each individual; use ggcontrib() from nadiv package
Q <- ggcontrib(d.ped[, 4:6], ggroups = c("inner", "outer","other"))
d.ped <- cbind(d.ped, Q)

# now copy this genetic group information into d.morph
d.morph$inner <- d.ped[match(d.morph$id,d.ped$id),"inner"]
d.morph$outer <- d.ped[match(d.morph$id,d.ped$id),"outer"]
d.morph$other <- d.ped[match(d.morph$id,d.ped$id),"other"]


# to find out how many animals had inner/outer/other natal island
d.tmp <- d.natal[d.natal$ringnr %in% d.morph$ringnr,]
table(d.tmp$islandIO )

