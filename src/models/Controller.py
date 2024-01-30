"""
Path: src/models/controller.py
This file is the controller of the model training process.
Reads the config file and runs the appropriate model.
"""
from .utils import (
    prep_data_before_train,
    handle_yaml_before_train,
    Dataset,
    ModelConfig,
)
from .ModelTrainer import ModelTrainer
from .ModelCV import ModelCV, ModelOuterInner, ModelAcrossIsland, ModelINLA
from pathlib import Path


def main():
    project_path = Path(__file__).resolve().parents[2]
    yaml_path = project_path / "config.yaml"
    modelSettings = ModelConfig(project_path, yaml_path)
    data_path = modelSettings.data_path

    if modelSettings.model == "INLA":
        modelCVobj = ModelINLA(data_path, modelSettings)
    elif modelSettings.train_across:
        try:
            if modelSettings.train_across_islands:
                modelCVobj = ModelAcrossIsland(data_path, modelSettings)
            else:
                modelCVobj = ModelOuterInner(data_path, modelSettings)
        except Exception as e:
            modelCVobj = ModelAcrossIsland(data_path, modelSettings)
    else:
        modelCVobj = ModelCV(data_path, modelSettings)

    modelCVobj.run()


if __name__ == "__main__":
    main()
