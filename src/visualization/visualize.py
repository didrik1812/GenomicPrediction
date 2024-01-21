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
RESULTS_FROM_PROJECT_DIR  = PROJECT_DIR/"data"/"external" 
SAVE_FIGURE_PATH = PROJECT_DIR/"reports"/"figures"
# LOAD DATAFRAMES
results_df = pd.read_pickle(PROJECT_DIR/"models"/"results.pkl")
project_thesis_result_df = pd.read_pickle(RESULTS_FROM_PROJECT_DIR/"project_df.pkl")
project_thesis_result_df_BV = pd.read_pickle(RESULTS_FROM_PROJECT_DIR/"project_df_BV.pkl")
project_thesis_result_df_EG = pd.read_pickle(RESULTS_FROM_PROJECT_DIR/"project_df_EG.pkl")

def compare_with_project():
    pass   

def viz_across_pop():
    across_df = results_df[results_df.fold.isin(["outer","inner"])]
    across_df = across_df.rename(columns={"model_id":"model"})
    sns.set_style("whitegrid")
    colors = ["#1f78b4", "#a6cee3","#b2df8a", "#fb9a99", "#fffff3"]
    sns.set_palette(sns.color_palette(colors))
    plt.figure()
    plt.title("Across Population Predictions")
    ax = sns.catplot(data = across_df, kind = "bar", x="phenotype", y = "corr", hue ="model",
        col = "fold", )
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(10))
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel("Correlation")
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(SAVE_FIGURE_PATH / "across_pop.pdf")



def make_boxplot(df:pd.DataFrame, title:str, fig_name:str):
    df = df.rename(columns={"model_id":"model"})
    sns.set_style("whitegrid")
    colors = ["#1f78b4", "#a6cee3","#b2df8a", "#fb9a99", "#ffff3"]
    sns.set_palette(sns.color_palette(colors))
    plt.figure()
    plt.title(title)
    ax = sns.boxplot(data=df, x="phenotype", y="corr", hue="model", orient="v", width=.8, showfliers=False)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(10))
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel("Correlation")
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(SAVE_FIGURE_PATH / fig_name)

if __name__ == "__main__":
    viz_across_pop()

