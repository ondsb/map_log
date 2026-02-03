import json
import os

import pandas as pd
from tqdm import tqdm
from config.train_dota2 import data_version

basedir = f"/home/user/ODDIN/llm/MjolnirMind/nanoGPT/out/dota2_{data_version}/matches_processed"


with open(
    "/home/user/ODDIN/llm/MjolnirMind/nanoGPT/data/dota2/hero_ids.json", "r"
) as readfile:
    hero_ids_lup = json.load(readfile)

for file in tqdm(os.listdir(basedir)):
    cur = pd.read_csv(f"{basedir}/{file}", index_col=0)
    m_id = file.split(".")[0]

    outs = []
    for i, grp in cur.groupby("i"):
        # check hero vs side
        kills = grp[grp.event == "kill"]
        ss = grp.iloc[-1]

        # kill distributions
        kills_light = kills[kills.side == "light"].shape[0]
        kills_dark = kills[kills.side == "dark"].shape[0]
        tk = kills_light + kills_dark

        # md dist
        md = ss.t

        # buildings to win
        bld = grp[grp.event == "building"]

        light_win = ss.side == "light"

        win_bld = bld[bld.side == ss.side]
        win_counts = win_bld.tar.value_counts()
        win_bld = win_counts.to_dict()

        loser = "light" if ss.side == "dark" else "dark"
        lose_bld = bld[bld.side == loser]
        lose_counts = lose_bld.tar.value_counts()
        lose_bld = lose_counts.to_dict()

        # out
        out = {
            "md": md,
            "tk": tk,
            "kills_light": kills_light,
            "kills_dark": kills_dark,
            "light_win": light_win,
            "win_bld": win_bld,
            "lose_bld": lose_bld,
        }

        outs.append(out)

    df = pd.DataFrame(outs)
    df.to_csv(f"{basedir}/../matches_stats/{file}")
