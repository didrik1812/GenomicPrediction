'''
Visualize Differences between the island groups
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from plotnine import (
    ggplot,
    aes,
    geom_boxplot,
    theme_bw,
    facet_grid
)


PROJECT_DIR = Path(__file__).resolve().parents[2]

def viz_between_outer_and_inner(df:pd.DataFrame, pheno:str):
    outer_idx = df.index[df.hatchisland.isin([22,23,24])]
    vgroup = np.vectorize(lambda x: "outer" if x in outer_idx else "inner")
    df["island_group"] = vgroup(df.index)
    months = df.month.unique()
    month_dict = {months[i]:str(i) for i in range(len(months))}
    vmonth = np.vectorize(lambda x: month_dict[x])
    df["month_group"] = vmonth(df.month)
    p = ggplot(df, aes(x="month_group", y ="pheno")) +\
            geom_boxplot()+\
            theme_bw()+\
            facet_grid(".~island_group")
    p.save(PROJECT_DIR/"reports"/"figures"/f"outer_inner_comp_{pheno}.pdf")
    

def main():
    TarsusEG_df = pd.read_feather(PROJECT_DIR/"data"/"processed"/"tarsusEG.feather") 
    MassEG_df = pd.read_feather(PROJECT_DIR/"data"/"processed"/"massEG.feather") 
    viz_between_outer_and_inner(TarsusEG_df, "tarsus") 
    viz_between_outer_and_inner(MassEG_df, "mass")

if __name__ == "__main__":
    main()
