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
RESULTS_FROM_PROJECT_DIR = PROJECT_DIR / "data" / "external"
SAVE_FIGURE_PATH = PROJECT_DIR / "reports" / "figures"
# LOAD DATAFRAMES
results_df = pd.read_pickle(PROJECT_DIR / "models" / "results.pkl")
project_thesis_result_df = pd.read_pickle(RESULTS_FROM_PROJECT_DIR /
                                          "project_df.pkl")
project_thesis_result_df_BV = pd.read_pickle(RESULTS_FROM_PROJECT_DIR /
                                             "project_df_BV.pkl")
project_thesis_result_df_EG = pd.read_pickle(RESULTS_FROM_PROJECT_DIR /
                                             "project_df_EG.pkl")

rename_dict = {
    "xgboostTarsusLinearEG": "xgboostLinear",
    "xgboostMasssLinearEG": "xgboostLinear",
    "xgboostTarsusLinearBV": "xgboostLinear"
}


def compare_with_project(names: list, fig_name: str, EG: bool = True):
    project_EG_red_df = project_thesis_result_df_EG.drop(
        columns=["MSE", "feat_perc", "corrWith", "EG"])
    project_BV_red_df = project_thesis_result_df_BV.drop(
        columns=["MSE", "feat_perc", "corrWith", "EG"])
    master_df = results_df[~results_df.fold.isin(["outer", "inner"])]
    master_df = master_df[master_df.name.isin(names)]
    master_df = master_df.rename(columns={
        "name": "model"
    }).drop(columns=["model_id", "fold"])
    master_df.model = master_df.model.replace(rename_dict)
    if EG:
        merged_df = pd.concat([project_EG_red_df, master_df], axis=0)
        title = "Phenotype Correlation"
    else:
        merged_df = pd.concat([project_BV_red_df, master_df],
                              ignore_index=True)
        title = "Breeding Value Correlation"
    make_boxplot(merged_df, title, fig_name)


def viz_across_pop():
    across_df = results_df[results_df.fold.isin(["outer", "inner"])]
    across_df = across_df.rename(columns={"model_id": "model"})
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
    )
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(10))
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.ylabel("Correlation")
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(SAVE_FIGURE_PATH / "across_pop.pdf")


def make_boxplot(df: pd.DataFrame, title: str, fig_name: str):
    sns.set_style("whitegrid")
    colors = ["#1f78b4", "#a6cee3", "#b2df8a", "#fb9a99", "#fffff3"]
    sns.set_palette(sns.color_palette(colors))
    plt.figure()
    plt.title(title)
    ax = sns.boxplot(data=df,
                     x="phenotype",
                     y="corr",
                     hue="model",
                     orient="v",
                     width=.8,
                     showfliers=False)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(10))
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.ylabel("Correlation")
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(SAVE_FIGURE_PATH / fig_name)


if __name__ == "__main__":
    viz_across_pop()
    compare_with_project(
        names=["xgboostTarsusLinearEG", "xgboostMasssLinearEG"],
        fig_name="EG_compare_with_linear.pdf")
    compare_with_project(names=["xgboostTarsusLinearBV","xgboostMasssLinearBV"],
                         fig_name="BV_compare_with_linear.pdf",
                         EG=False)
