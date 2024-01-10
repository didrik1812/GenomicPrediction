.PHONY: all, data, train

all: train INLA
train: data config.yaml
	python src/models/train_model.py


data: tarsusEG massEG tarsusBV massBV


massBV: data/processed/massBV.feather
tarsusBV: data/processed/tarsusBV.feather

data/processed/massBV.feather: src/data/dataloader.R
	Rscript --vanilla src/data/dataloader.R mass

data/processed/tarsusBV.feather: src/data/dataloader.R
	Rscript --vanilla src/data/dataloader.R tarsus

massEG: data/processed/massEG.feather
tarsusEG: data/processed/tarsusEG.feather

data/processed/massEG.feather: src/data/envGendataloader.R
	Rscript --vanilla src/data/envGendataloader.R mass

data/processed/tarsusEG.feather: src/data/envGendataloader.R
	Rscript --vanilla src/data/envGendataloader.R tarsus

# RUN INLA BY using "make INLA phenotype=mass"
INLA: data/interim
data/interim: src/models/INLAcv.R
	Rscript --vanilla src/models/INLAcv.R
