import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from ..features import searchspaces
import re
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
import json

PROJECT_DIR = Path(__file__).resolve().parents[2]
TYPE_OF_MODEL = "Linear"


def get_names_and_phenotype(modelName:str)->Tuple[str,str, bool]:
    NameSplitted = re.findall(r'[a-zA-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', modelName)
    phenotype = NameSplitted[1]
    is_wanted_model = bool(NameSplitted[2] == TYPE_OF_MODEL)
    if NameSplitted[2] in ["EGA", "EG", "BV"]:
        newName = NameSplitted[0]
    else:
        newName = NameSplitted[0] + NameSplitted[2]
  
    return [newName,phenotype, is_wanted_model]

def get_model_obj(modelName: str):
    models_dir = PROJECT_DIR/"models"/modelName
    models = list(models_dir.glob("*.json"))
    for model in models:
        modelObj = XGBRegressor()
        modelObj.load_model(model)
        b = modelObj.get_booster()
        configs = json.loads(b.save_config())["learner"]["gradient_booster"]["updater"]["linear_train_param"]
        yield configs
        # modelObj = pickle.load(open(model, "rb"))
        # yield modelObj
        # modelObj.save_model(model.parent/f"{model.stem}.json")

def plot_hyperparameters(df:pd.DataFrame, name:str)->None:
    num_plots = len(df.columns) - 2
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots))
    for i, col in enumerate(df.columns[:-2]):
        sns.boxplot(x="model", y=col, data=df, ax=axes[i])
        axes[i].set_title(f"{col} distribution")
        axes[i].set_xlabel("Model")
        axes[i].set_ylabel(col)
    plt.tight_layout()
    plt.savefig(PROJECT_DIR/"reports"/"figures"/f"hyperparameters_{name}.pdf")

def main():
    models_list = [p.name for p in (PROJECT_DIR/"models").iterdir() if p.is_dir()]
    model_pheno_list = [get_names_and_phenotype(model) for model in models_list]
    model_name_list = [model_pheno[0] for model_pheno in model_pheno_list if model_pheno[2]]
    pheno_list = [model_pheno[1] for model_pheno in model_pheno_list if model_pheno[2]]
    models_list  = [m for i,m in enumerate(models_list) if model_pheno_list[i][2]]


    params_to_plot = list( searchspaces.xgboost_linear_space.keys())

    dict_list = []
    print(models_list)
    for i in range(len(models_list)):
        if models_list[i] == "xgboostMasssLinear_quantEGA": 
            continue
        for config in get_model_obj(models_list[i]):
            model_params_to_keep = {k: config.get(k, 0) for k in params_to_plot}
            model_params_to_keep["model"] = model_name_list[i] 
            model_params_to_keep["phenotype"] = pheno_list[i]
            dict_list.append(model_params_to_keep)

    plot_df = pd.DataFrame.from_records(dict_list) 
    print(plot_df)
    plot_hyperparameters(plot_df,TYPE_OF_MODEL)

if __name__ == "__main__":
    main()
