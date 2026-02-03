import json
import os

import pandas as pd

from config.train_dota2 import data_version
from model.scaler import Scaler, ScalerPars

basedir = (
    f"/home/user/ODDIN/llm/MjolnirMind/nanoGPT/out/dota2_{data_version}/matches_stats"
)
out_dir = f"/home/user/ODDIN/llm/MjolnirMind/nanoGPT/out/dota2_{data_version}/val"

maps = f"/home/user/ODDIN/llm/MjolnirMind/nanoGPT/data/dota2/maps_{data_version}.json"

status = pd.read_hdf("/home/user/ODDIN/data/dota2/dota2_all.h5", "status")
team_stats = pd.read_hdf("/home/user/ODDIN/data/dota2/dota2_all.h5", "team_stats")

with open(maps, "r") as openfile:
    maps = json.load(openfile)

gt_scaler = Scaler(ScalerPars(-3, 3, 0, 6500))
tk_scaler = Scaler(ScalerPars(-3, 3, 0, 150))
pr_scaler = Scaler(ScalerPars(-3, 3, 0, 1))

big = []
for file in os.listdir(basedir):
    map_id = file.split(".")[0]
    cur = pd.read_csv(f"{basedir}/{file}", index_col=0)
    cur_status = status[status.oddin_map_id == int(map_id)].iloc[0]
    cur_team_stats = team_stats[team_stats.oddin_map_id == int(map_id)].iloc[0]

    stats = pd.read_csv(f"{basedir}/{file}", index_col=0)

    obs = maps[map_id].split(">")

    prematch = pr_scaler.inverse_transform(float(obs[3].split(" ")[2]))
    lw_obs = abs(cur_status.oddin_home_is_dark - cur_status.oddin_home_win)
    lw_mod = cur.light_win.mean()

    dur_obs = cur_status.duration
    dur_mod = cur.md.median()

    tk_obs = cur_team_stats.kill_score_light + cur_team_stats.kill_score_dark
    tk_mod = cur.tk.median()

    kl_obs = cur_team_stats.kill_score_light
    kl_mod = cur.kills_light.median()
    kd_obs = cur_team_stats.kill_score_dark
    kd_mod = cur.kills_dark.median()

    kdiff_obs = kl_obs - kd_obs
    kdiff_mod = kl_mod - kd_mod

    big.append(
        {
            "prematch": prematch,
            "lw_obs": lw_obs,
            "lw_mod": lw_mod,
            "dur_obs": dur_obs,
            "dur_mod": dur_mod,
            "tk_obs": tk_obs,
            "tk_mod": tk_mod,
            "kdiff_obs": kdiff_obs,
            "kdiff_mod": kdiff_mod,
        }
    )

big = pd.DataFrame(big)
big.to_csv(f"{out_dir}/val.csv")
