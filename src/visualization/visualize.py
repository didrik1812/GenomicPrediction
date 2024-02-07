'''
Vizualization script, compares models to one eachother.
'''
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import matplotlib.ticker as ticker
from ..models import utils
import re
# GLOBAL VARIABLES
PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_FROM_PROJECT_DIR = PROJECT_DIR / "data" / "external"
SAVE_FIGURE_PATH = PROJECT_DIR / "reports" / "figures"
# LOAD DATAFRAMES
results_df = pd.read_pickle(PROJECT_DIR / "models" / "results.pkl")
project_thesis_result_df = pd.read_pickle(RESULTS_FROM_PROJECT_DIR / "project_df.pkl")
project_thesis_result_df_BV = pd.read_pickle(RESULTS_FROM_PROJECT_DIR / "project_df_BV.pkl")
project_thesis_result_df_EG = pd.read_pickle(RESULTS_FROM_PROJECT_DIR / "project_df_EG.pkl")


def rename_models(oldName: str):
    # Split oldName (camelcase) to list using regex
    NameSplitted = re.findall(r'[a-zA-Z](?:[a-z_]+|[A-Z]*(?=[A-Z]|$))', oldName)

    if NameSplitted[2] in ["EGA", "EG", "BV"]:
        newName = NameSplitted[0]
    else:
        newName = NameSplitted[0] + NameSplitted[2]
    return newName


def compare_with_project(names: list, fig_name: str, EG: bool = True):
    project_EG_red_df = project_thesis_result_df_EG.drop(columns=["MSE", "feat_perc", "corrWith", "EG"])
    project_BV_red_df = project_thesis_result_df_BV.drop(columns=["MSE", "feat_perc", "corrWith", "EG"])
    master_df = results_df[~results_df.fold.isin(["outer", "inner"])]
    master_df = master_df[master_df.name.isin(names)]
    master_df = master_df.rename(columns={"name": "model"}).drop(columns=["model_id", "fold"])
    master_df.model = master_df.model.apply(rename_models)
    if EG:
        merged_df = pd.concat([project_EG_red_df, master_df], axis=0)
        title = "Phenotype Correlation"
    else:
        merged_df = pd.concat([project_BV_red_df, master_df], ignore_index=True)
        title = "Breeding Value Correlation"
    make_boxplot(merged_df, title, fig_name)


def viz_across_pop():
    across_df = results_df[results_df.fold.isin(["outer", "inner"])]
    across_df = across_df.dropna(subset = ["corr"])
    across_df = across_df.rename(columns={"name": "model"})
    across_df.model = across_df.model.apply(rename_models)
    sns.set_style("whitegrid")
    colors = ["#1f78b4", "#a6cee3", "#b2df8a", "#fb9a99", "#fffff3"]
    sns.set_palette(sns.color_palette(colors))
    plt.figure()
    plt.title("Across Population Predictions")
    ax = sns.catplot(
        data=across_df,
        kind="bar",
        x="phenotype",
        y="corr",
        hue="model",
        col="fold",
        legend=True,
    )
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(10))
    # ax.despine(left=True)
    # plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.ylabel("Correlation")
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(SAVE_FIGURE_PATH / "across_pop.pdf")

def viz_island_for_island():
    island_dfs = results_df[results_df.fold.isin[]]

def make_boxplot(df: pd.DataFrame, title: str, fig_name: str):
    sns.set_style("whitegrid")
    colors = ["#1f78b4", "#a6cee3", "#b2df8a", "#fb9a99", "#fffff3"]
    sns.set_palette(sns.color_palette(colors))
    plt.figure()
    plt.title(title)
    ax = sns.boxplot(data=df, x="phenotype", y="corr", hue="model", orient="v", width=.8, showfliers=False)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(10))
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.ylabel("Correlation")
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(SAVE_FIGURE_PATH / fig_name)


if __name__ == "__main__":
    BVnames, EGnames, AcrossPopNames = utils.get_current_model_names()
    viz_across_pop()
    compare_with_project(names=EGnames, fig_name="EG_compare_with_linear.pdf")
    compare_with_project(names=BVnames, fig_name="BV_compare_with_linear.pdf", EG=False)
