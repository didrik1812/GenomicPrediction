# Configs
This folder contains the config files that define the models used in the Master thesis. 
It contains four subfolders that each correspons to the four box-plots showed in the results section in the thesis.
* within70K: models trained for within-population predictions on 70K dataset
* within180K: models trained for within-population predictions on 180K dataset, note that some models were from the project thesis and are therefore not here, but in the [projectThesis](https://github.com/didrik1812/ProjectThesis) repo
* acrossBaseLearner: models trained for across-population predictions on 70K dataset
*  acrossBaseLoss: gradient boosting with piecewise linear regression trees trained for across-population predictions on 70K dataset with different loss functionst


So if you want to train a tree-based XGBoost model for body mass on the 70K dataset, go into the `within70K/xgboostMass.yaml` file and copy the contents into `GenomicPrediction/config.yaml` file and run `make` in the terminal. 