'''
Vizualization script, compares models to one eachother.
'''
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import matplotlib.ticker as ticker
# GLOBAL VARIABLES
PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULSTS_FROM_PROJECT_DIR  = PROJECT_DIR/"data"/"external" 
SAVE_FIGURE_PATH = PROJECT_DIR/"reports"/"figures"
# LOAD DATAFRAMES
results_df = pd.read_feather(PROJECT_DIR/"models"/"result.feather")
project_thesis_result_df = pd.read_pickle(RESULTS_FROM_PROJECT_DIR/"project_df.pkl")
project_thesis_result_df_BV = pd.read_pickle(RESULTS_FROM_PROJECT_DIR/"project_df_BV.pkl")
project_thesis_result_df_EG = pd.read_pickle(RESULTS_FROM_PROJECT_DIR/"project_df_EG.pkl")

def compare_with_project():
    pass

def make_viz(df:pd.DataFrame):
    pass
