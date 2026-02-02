import ast
import json
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from nanoGPT.model.scaler import Scaler, ScalerPars
from nanoGPT.config.train_dota2 import data_version

srcdir = (
    f"/home/user/ODDIN/llm/MjolnirMind/nanoGPT/out/dota2_{data_version}/matches_runs"
)
basedir = (
    f"/home/user/ODDIN/llm/MjolnirMind/nanoGPT/out/dota2_{data_version}/matches_stats"
)
out_dir = f"/home/user/ODDIN/llm/MjolnirMind/nanoGPT/out/dota2_{data_version}/fig"

gt_scaler = Scaler(ScalerPars(-3, 3, 0, 6500))
tk_scaler = Scaler(ScalerPars(-3, 3, 0, 150))
pr_scaler = Scaler(ScalerPars(-3, 3, 0, 1))


big = []
for file in tqdm(os.listdir(basedir)):
    pass

    mid = file.split(".")[0]
    with open(f"""{srcdir}/run_{mid}.json""") as openfile:
        src = json.load(openfile)

    prem = src[mid][0].split(">")[3].split(" ")
    prematch, dur, tk = (
        pr_scaler.inverse_transform(float(prem[2])),
        gt_scaler.inverse_transform(float(prem[6])),
        tk_scaler.inverse_transform(float(prem[8])),
    )

    df = pd.read_csv(f"{basedir}/{file}", index_col=0)
    df["kdiff"] = df.kills_light - df.kills_dark

    # mdtk
    sns.histplot(df, x="md", y="tk", cbar=True)
    plt.title(np.corrcoef(df.md, df.tk)[0][1])
    plt.savefig(f"""{out_dir}/{file.split(".")[0]}_mdtk.pdf""")
    plt.close()

    # kdiff
    lw_df = df[df.light_win]
    sns.histplot(lw_df, x="kdiff", bins=15)
    dw_df = df[~df.light_win]
    sns.histplot(dw_df, x="kdiff", bins=15)
    plt.savefig(f"""{out_dir}/{file.split(".")[0]}_kdiff.pdf""")
    plt.close()

    # tow
    n_tow_w = df.win_bld.agg(
        lambda x: sum([v for k, v in ast.literal_eval(x).items() if "tower" in k])
    )
    n_tow_l = df.lose_bld.agg(
        lambda x: sum([v for k, v in ast.literal_eval(x).items() if "tower" in k])
    )
    sns.histplot(n_tow_w, bins=15)
    sns.histplot(n_tow_l, bins=15)
    plt.savefig(f"""{out_dir}/{file.split(".")[0]}_tow.pdf""")
    plt.close()

    # rax
    n_rax_w = df.win_bld.agg(
        lambda x: sum([v for k, v in ast.literal_eval(x).items() if "barracks" in k])
    )
    n_rax_l = df.lose_bld.agg(
        lambda x: sum([v for k, v in ast.literal_eval(x).items() if "barracks" in k])
    )
    sns.histplot(n_rax_w)
    sns.histplot(n_rax_l)
    plt.savefig(f"""{out_dir}/{file.split(".")[0]}_barracks.pdf""")
    plt.close()

    big.append(
        {
            "ml": df.light_win.values.mean(),
            "ml_prematch": prematch,
            "md": df.md.values,
            "md_prematch": dur,
            "tk": df.tk.values,
            "tk_prematch": tk,
            "kdiff_lw": lw_df.kdiff.values,
            "kdiff_dw": dw_df.kdiff.values,
            "tow_w": n_tow_w.values,
            "tow_l": n_tow_l.values,
            "rax_w": n_rax_w.values,
            "rax_l": n_rax_l.values,
        }
    )

b = pd.DataFrame(big)

# mdtk
md_all = b.md.agg(np.concatenate)
tk_all = b.tk.agg(np.concatenate)
sns.histplot(x=md_all, y=tk_all, cbar=True)
plt.title(np.corrcoef(md_all, tk_all)[0][1])
plt.savefig(f"""{out_dir}/_mdtk.pdf""")
plt.close()

# kdiff
kdiff_lw_all = b.kdiff_lw.agg(np.concatenate)
kdiff_dw_all = b.kdiff_dw.agg(np.concatenate)
sns.histplot(kdiff_lw_all, bins=15)
sns.histplot(kdiff_dw_all, bins=15)
plt.savefig(f"""{out_dir}/_kdiff.pdf""")
plt.close()

# tow
n_tow_w = b.tow_w.agg(np.concatenate)
n_tow_l = b.tow_l.agg(np.concatenate)
sns.histplot(n_tow_w, bins=15)
sns.histplot(n_tow_l, bins=15)
plt.savefig(f"""{out_dir}/_tow.pdf""")
plt.close()

# rax
n_rax_w = b.rax_w.agg(np.concatenate)
n_rax_l = b.rax_l.agg(np.concatenate)
sns.histplot(n_rax_w)
sns.histplot(n_rax_l)
plt.savefig(f"""{out_dir}/_rax.pdf""")
plt.close()

# prematch
sc = sns.scatterplot(x=b.ml_prematch.values, y=b.ml.values)
diagonal = plt.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
plt.savefig(f"{out_dir}/_prematch.pdf")
plt.close()

# md
md = sns.scatterplot(x=b.md_prematch.values, y=b.md.apply(lambda x: np.mean(x)).values)
diagonal = plt.plot(
    [0, b.md_prematch.max()],
    [0, b.md_prematch.max()],
    color="black",
    linestyle="--",
    linewidth=1,
)
plt.savefig(f"{out_dir}/_md.pdf")
plt.close()

# tk
tk = sns.scatterplot(x=b.tk_prematch.values, y=b.tk.apply(lambda x: np.mean(x)).values)
diagonal = plt.plot(
    [0, b.tk_prematch.max()],
    [0, b.tk_prematch.max()],
    color="black",
    linestyle="--",
    linewidth=1,
)
plt.savefig(f"{out_dir}/_tk.pdf")
plt.close()
