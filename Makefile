.PHONY: all, data, train, predict

massBV: data/processed/massBV.feather
tarsusBV: data/processed/tarsusBV.feather

massEG: data/processed/massEG.feather
tarsusEG: data/processed/tarsusEG.feather

data/processed/massBV.feather: src/data/dataloader.R
	Rscript --vanilla src/data/dataloader.R mass

data/processed/tarsusBV.feather: src/data/dataloader.R
	Rscript --vanilla src/data/dataloader.R tarsus

massEG: data/processed/massBV.feather
tarsusEG: data/processed/tarsusBV.feather

data/processed/massEG.feather: src/data/envGendataloader.R
	Rscript --vanilla src/data/envGendataloader.R mass

data/processed/tarsusEG.feather: src/data/envGendataloader.R
	Rscript --vanilla src/data/envGendataloader.R tarsus