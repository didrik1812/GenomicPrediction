from pathlib import Path
from .utils import (
    ModelConfig,
)
from .modules import ModelCV as mcv
from pathlib import Path
import numpy as np



project_path = Path(__file__).resolve().parents[2]
print(project_path)
yaml_path = project_path / "config.yaml"
modelSettings = ModelConfig(project_path, yaml_path)
data_path = modelSettings.data_path
modelCVobj = mcv.ModelOuterInner
modelCVobj = type("ModelCVQuantile", (modelCVobj,), {"train_and_eval": mcv.train_and_eval_quantile})
print(modelSettings.name)
modelCVinstance = modelCVobj(data_path, modelSettings)
print(modelCVinstance.project_path)
methods = [method for method in dir(modelCVinstance) if method.startswith("__")
           is False]
print(methods)



# massBV_df = pd.read_feather("data/processed/massEG.feather")
# print(massBV_df.iloc[:,12:50].dtypes)
# print(massBV_df.isna().sum().head(15))
# num_outer_inds = massBV_df.outer.value_counts()[1.0]
# print(len(massBV_df.ringnr.unique()))
# print(num_outer_inds)
# print(massBV_df.columns[:15])
# SNP_cols = list(massBV_df.columns[8:])
# positions = [s.split("SNP")[1].split("_")[0] for s in SNP_cols]
# new_pos = [0]*len(positions)
# for i in range(len(positions)):
#     try:
#         new_pos[i] = positions[i].split("a")[1]
#     except:
#         new_pos[i] = positions[i].split("a")[0]
#     try:
#         new_pos[i] = new_pos[i].split("i")[1]
#     except:
#         new_pos[i]=new_pos[i].split("i")[0]
# new_pos = pd.Series(new_pos, dtype = float)

# print(new_pos[:20])
# pl.hist(new_pos.to_numpy(),norm=True,bins = 200)
# pl.show()
