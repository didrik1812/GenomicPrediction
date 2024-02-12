"""
Path: src/models/train_model.py
This file is the controller of the model training process.
Reads the config file and runs the appropriate model.

About the setup:
    Currently three possible setups are possible:
    1. Train a model with cross-validation (CV) using the ModelCV class.
        This uses the entire dataset (i.e. no seperation between islands)
    2. Train a model with CV using the ModelOuterInner class.
        This uses the entire dataset, but splits the data into outer and inner islands.
        The model is trained on one island and tested on the other.
        This is used for a mild version of across population predictions.
    3. Train a model with CV using the ModelAcrossIsland class.
        This uses the entire dataset, but splits the data into islands.
        The model is trained on all islands except one and tested on the left out island.
        This is used for to get more datapoints of the models performance for across population predictions.
"""
from .utils import (
    ModelConfig,
)
from .modules import ModelCV as mcv
from pathlib import Path
import numpy as np

np.random.seed(42)
# TODO: Make dict to replace the if-else mess in main()

def main():
    project_path = Path(__file__).resolve().parents[2]
    yaml_path = project_path / "config.yaml"
    modelSettings = ModelConfig(project_path, yaml_path)
    data_path = modelSettings.data_path

    if modelSettings.model == "INLA":
        modelCVobj = mcv.ModelINLA
    elif modelSettings.train_across_islands:
        modelCVobj = mcv.ModelAcrossIsland
    elif modelSettings.train_across:
        modelCVobj = mcv.ModelOuterInner
    else:
        modelCVobj = mcv.ModelCV

    if "reg:quantileerror" in modelSettings.fixed_params.values():
        # dynamically create a ModelCVQuantile class with the suitable train_and_eval method for quantrile regression.
        modelCVobj = type("ModelCVQuantile", (modelCVobj,), {"train_and_eval": mcv.train_and_eval_quantile, "add_to_results": mcv.add_to_results_quantile})

    modelCVinstance = modelCVobj(data_path, modelSettings)
    modelCVinstance.run()


if __name__ == "__main__":
    main()
