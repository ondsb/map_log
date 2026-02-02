import json
import os

import pandas as pd
from tqdm import tqdm

from nanoGPT.model.scaler import Scaler, ScalerPars
from nanoGPT.config.train_dota2 import data_version

basedir = (
    f"/home/user/ODDIN/llm/MjolnirMind/nanoGPT/out/dota2_{data_version}/matches_runs/"
)

runs = [x for x in os.listdir(basedir) if "run" in x]

gt_scaler = Scaler(ScalerPars(-3, 3, 0, 6500))
tk_scaler = Scaler(ScalerPars(-3, 3, 0, 150))
pr_scaler = Scaler(ScalerPars(-3, 3, 0, 1))


for run in runs:
    print()
    print(run)
    with open(f"{basedir}/{run}", "r") as readfile:
        run = json.load(readfile)

    match_id = list(run.keys())[0]

    big = []
    for i, out in enumerate(tqdm(run[match_id])):
        prompt = out.split(">")[3].split(" ")
        prematch, prematch_dur, prematch_tk = (
            pr_scaler.inverse_transform(float(prompt[2])),
            gt_scaler.inverse_transform(float(prompt[6])),
            tk_scaler.inverse_transform(float(prompt[8])),
        )

        for cz in out.split(">")[4:]:
            # print(cz)

            try:
                cz = cz.split(" ")[1:]

                if cz[2] == "killed":
                    event = "kill"
                elif cz[2] == "destroyed":
                    event = "building"
                elif cz[2] == "slayed":
                    event = "npc"
                else:
                    raise Exception("bad")

                side = cz[0]
                t = gt_scaler.inverse_transform(float(cz[5]))
                att = cz[1]
                tar = cz[3]

                c_out = dict(
                    match_id=match_id,
                    i=i,
                    t=t,
                    event=event,
                    side=side,
                    att=att,
                    tar=tar,
                )
                big.append(c_out)
            except Exception as e:
                print(e)

    big = pd.DataFrame(big, columns=list(c_out.keys()))
    big.to_csv(f"{basedir}/../matches_processed/{match_id}.csv")
