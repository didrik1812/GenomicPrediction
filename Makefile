.PHONY: all tarsusEG massEG tarsusBV massBV train INLA fig

all: train INLA fig
train: data config.yaml
	nice python -m src.models.train_model

data: tarsusEG massEG tarsusBV massBV


massBV: data/processed/massBV.feather

tarsusBV: data/processed/tarsusBV.feather

data/processed/massBV.feather: src/data/dataloader.R
	nice Rscript --vanilla src/data/dataloader.R mass

data/processed/tarsusBV.feather: src/data/dataloader.R
	nice Rscript --vanilla src/data/dataloader.R tarsus

massEG: data/processed/massEG.feather

tarsusEG: data/processed/tarsusEG.feather

data/processed/massEG.feather: src/data/envGendataloader.R
	nice Rscript --vanilla src/data/envGendataloader.R mass

data/processed/tarsusEG.feather: src/data/envGendataloader.R
	nice Rscript --vanilla src/data/envGendataloader.R tarsus

# RUN INLA BY using "make INLA phenotype=mass"
INLA: data/interim
data/interim: src/models/INLAcv.R
	nice Rscript --vanilla src/models/INLAcv.R

fig:
	nice python -m src.visualization.visualize
