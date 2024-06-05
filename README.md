# Master Thesis - Genomic Prediction
This repo contains code related to my master thesis where new ways of performing genomic prediction is explored. 

## File Structure

The organization is inspired by the first version of the [Cookie Cutter for Data Science](https://cookiecutter-data-science.drivendata.org/v1/). The file tree is organized as follows


```
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── config.yaml        <- Config file for training, ideally the only file that needs to be changed between models   
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Here we typically stored data from the project thesis
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- Here data was stored after merging the adjusted phenotype with the SNP data
│   └── raw            <- Here we stored the original morphological and genomic data we recived.
│
├── models             <- Trained models. The results dataframe is also stored here here. Each model is stored in its own dir
│   ├── model1         <- A directory for one model, includes the stored model and the config file for that model 
|
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── environment.yml   <- The requirements file for reproducing the analysis environment
│
├── setup.py          <- Make this project pip installable with `pip install .` or `pip install -e`
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts generate and prep data
│   │   └── dataloader.R <- Main script for creating the adjusted phenotype and merge it with SNP data
|   |   └── h_dataPrep.r <- Helper script for the dataloader script
|   |   └── qc_bash.sh   <- Prep SNP data by a quality controll with PLINK
|   |   └── envGendataloader <- A version of the dataloader where the phenotype is not adjusted for environmental effects.
│   │
│   ├── features       <- Scripts to specify different features of the model
│   │   └── search_spaces.py <- Define search spaces for the different models
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   └── train_model.py <- Controller script for training models, reads the config.yaml and starts training (and testing)
│   │   └── get_shap.py <- Obtain SHAP values 
│   │   └── INLAcv.py   <- Run Bayesian animal model using INLA in a cross-validation
│   │   └── utils.py    <- Small helper functions and classes to handle data
│   |   └── modules     <- Contain classes/modules to run cross-validations and hyperparameter tuning
|   |           |
|   |           └── ModelCV.py <- Controlls the cross-validation/Island-for-Island procedure
|   |           └── ModelTrainer.py <- Peforms hyperparameter tuning and trains a model, used by ModelCV.py
│   │
│   └── visualization  <- Scripts to script to make manhatten plots of SHAP and GWAS
│       └── prepGWAS.sh <- Main script, this is the only script that needs to be alterd and run.
|       └── prepGWAS.R <- Prep some data before running GEMMA
|       └── GWAS.R      <- Make manhatten plot of GWAS results and compare with SHAP
|       └── manhatten.R <- Make manhatten plot of SHAP values
```

## How to replicate results
**Requirements:** Python (version 3.8+) , anaconda and R (version 4.3+)

First set up the environment by: 
1. Clone the repo
2. Create conda environment GenomicPrediction with necessary packages by running conda env create -f environment.yml.
3. Activate the environment `conda activate GenomicPrediction`
3. Install the repo by `pip install .`

### Model Training
Chose one of the config file of your desired model from the `configs/` folder and copy it into `config.yaml`. Then being in folder GenomicPrediction run `make` in the terminal. Results are stored in the `results.pkl` file under `models/`

### GWAS and SHAP
First make sure you have a tree based model trained. Then follow these step 
1. Go into `src/models/get_shap.py` and modify the `mod_path` variable such that it corresponds to the model you want to explain.
2. Run `python -m src.models.get_shap` to obtain SHAP values
3. Modify `src/visualization/prepGWAS.sh` (described in file) so that it corresponds to your models and the phenotype of intrest
4. Run  `sh prepGWAS.sh`
5. Plots are stored under `reports/figures`
