import json
import math
import os
import pickle
import time

import pandas as pd
from tqdm import tqdm

from nanoGPT.model.scaler import Scaler, ScalerPars

DATA_PATH = "/home/user/ODDIN/data/dota2"
OUT_DATA_PATH = "/home/user/ODDIN/llm/MjolnirMind/nanoGPT/data/dota2"
MAPS_OUT = "/home/user/ODDIN/llm/MjolnirMind/nanoGPT/data/dota2/maps_v7.json"


data_path = f"{DATA_PATH}/dota2_all.h5"
snapshots_cols = [
    "oddin_map_id",
    "game_time",
    "barracks_melee_alive_light",
    "barracks_range_alive_light",
    "barracks_melee_alive_dark",
    "barracks_range_alive_dark",
]

status = pd.read_hdf(data_path, "status")
player_kills = pd.read_hdf(data_path, "player_kills")
team_stats = pd.read_hdf(data_path, "team_stats")
building_kills = pd.read_hdf(data_path, "building_kills")
neutral_kills = pd.read_hdf(data_path, "neutral_kills")
picks = pd.read_hdf(data_path, "pick_and_ban")
snapshots = pd.read_hdf(data_path, "snapshots")[snapshots_cols]

status = status.dropna(subset="prematch_home")


with open(f"{OUT_DATA_PATH}/hero_ids.json", "r") as readfile:
    hero_ids = json.load(readfile)

# todo: find barracks since incorrect in building kills
barracks_snapshots = {}
for _, df in tqdm(snapshots.groupby("oddin_map_id")):
    cur_out = {}
    for team in ["light", "dark"]:
        opponent = "light" if team == "dark" else "dark"
        for barracks in ["barracks_melee_alive_", "barracks_range_alive_"]:
            df[f"inc_{barracks}{team}"] = (
                df[f"{barracks}{team}"].shift() != df[f"{barracks}{team}"]
            )
            kill_times = df[df[f"inc_{barracks}{team}"]].iloc[1:].game_time.to_list()
            cur_out[(opponent, barracks.split("_alive")[0][9:])] = kill_times
    barracks_snapshots[df.oddin_map_id.values[0]] = cur_out


# todo: norm
# GT_NORM = 6422  # max 6422 status.duration.values.max()
# SCORE_NORM = 80  # max 80
# TOWER_NORM = 4  # max 12?
# BARRACKS_NORM = 2  # max 12?

gt_scaler = Scaler(ScalerPars(-3, 3, 0, 6500))
tk_scaler = Scaler(ScalerPars(-3, 3, 0, 150))
pr_scaler = Scaler(ScalerPars(-3, 3, 0, 1))


def extract_dota2_v1():
    maps = {}
    for cur_map_id in tqdm(status.oddin_map_id.unique()):
        try:
            cur_st = status[status.oddin_map_id == cur_map_id]

            if pd.to_datetime(cur_st.date).dt.year.values[0] < 2022 and (
                cur_st.prematch_only.values[0]
                or cur_st.oddin_home_is_dark.values[0] == -1
                or not cur_st.prematch_home.values[0]
            ):
                continue

            cur_pk = player_kills[player_kills.oddin_map_id == cur_map_id]
            cur_bk = building_kills[building_kills.oddin_map_id == cur_map_id]
            cur_nk = neutral_kills[neutral_kills.oddin_map_id == cur_map_id]
            cur_pick = picks[(picks.oddin_map_id == cur_map_id) & (picks.action == 1)]
            cur_barracks = barracks_snapshots[cur_map_id]

            team_light = (
                cur_st.valve_light_team_name.values[0].replace(" ", "_").lower()
            )
            team_dark = cur_st.valve_dark_team_name.values[0].replace(" ", "_").lower()
            tournament = (
                cur_st.oddin_tournament_name.values[0].replace(" ", "_").lower()
            )
            tier = cur_st.tier.values[0]
            hid = cur_st.oddin_home_is_dark.values[0]
            prematch_light = abs(hid - float(cur_st.prematch_home.values[0]))
            heroes_light = [
                hero_ids[str(x.valve_hero_id)]
                for _, x in cur_pick.iterrows()
                if x.is_light
            ]
            heroes_dark = [
                hero_ids[str(x.valve_hero_id)]
                for _, x in cur_pick.iterrows()
                if not x.is_light
            ]

            def get_pk_dict(cur_pk):
                out = {}
                score = {"light": 0, "dark": 0}
                for i in range(cur_pk.shape[0]):
                    pk = cur_pk.iloc[i]
                    attacker = hero_ids[str(int(pk.attacker_hero_id))]
                    att_side = (
                        "light"
                        if cur_pick[
                            cur_pick.valve_hero_id == int(pk.attacker_hero_id)
                        ].is_light.values[0]
                        else "dark"
                    )
                    score[att_side] += 1
                    tar_str = (
                        f"{att_side} {attacker} killed {hero_ids[str(int(pk.target_hero_id))]}"
                        f""" at {gt_scaler.transform(int(pk.game_time))} """
                        # score {sc_scaler.transform(score["light"])} : {sc_scaler.transform(score["dark"])}
                    )
                    if int(pk.game_time) not in out:
                        out[int(pk.game_time)] = tar_str
                    else:
                        out[int(pk.game_time)] += tar_str
                        # print(out[int(pk.game_time)])
                return out

            def get_bk_dict(cur_bk, cur_barracks, cur_st):
                light_win = abs(
                    int(cur_st.oddin_home_is_dark.iloc[0])
                    - int(cur_st.oddin_home_win.iloc[0])
                )
                out = {}
                n_towers = {"light": 0, "dark": 0}
                for i in range(cur_bk.shape[0]):
                    bk = cur_bk.iloc[i]
                    side = "light" if bk.attacker_is_light else "dark"
                    side_str = "light" if light_win else "dark"
                    n_towers[side] += 1
                    tar_str = (
                        f"""{side if bk.building_type != "nexus" else side_str} team destroyed """
                        # f"""{n_towers[side] / TOWER_NORM if bk.building_type != "nexus" else 1} """
                        f"{bk.building_type} at {gt_scaler.transform(int(bk.game_time))} "
                    )
                    if int(bk.game_time) not in out:
                        out[int(bk.game_time)] = tar_str
                    else:
                        out[int(bk.game_time)] += tar_str
                        # print(out[int(bk.game_time)])

                for k, v in cur_barracks.items():
                    for i, gt in enumerate(v):
                        tar_str = f"{k[0]} team destroyed {k[1]}_barracks at {gt_scaler.transform(int(gt))} "
                        if gt not in out:
                            out[gt] = tar_str
                        else:
                            out[gt] = tar_str + "> " + out[gt]
                            # print(out[gt])

                return out

            def get_nk_dict(cur_nk):
                out = {}
                for i in range(cur_nk.shape[0]):
                    nk = cur_nk.iloc[i]
                    side = "light" if nk.attacker_is_light else "dark"
                    out[
                        int(nk.game_time)
                    ] = f"{side} team slayed {nk.target_type} at {gt_scaler.transform(int(nk.game_time))} "
                return out

            pk_dict = get_pk_dict(cur_pk)
            bk_dict = get_bk_dict(
                cur_bk[cur_bk.building_type != "barracks"],
                cur_barracks,
                cur_st,
            )
            nk_dict = get_nk_dict(cur_nk)
            sorted_events = sorted((pk_dict | bk_dict | nk_dict).items())

            tk = len(pk_dict)
            dur = sorted_events[-1][0]
            win_team = sorted_events[-1][1].split(" ")[0]
            win_team = (
                float(win_team == "light") if win_team in ["light", "dark"] else None
            )
            if win_team is None:
                print("incorrect map winner")
            prematch = f"""{tier} > {team_light} light {" ".join(heroes_light)} > {team_dark} dark {" ".join(heroes_dark)} > prematch {pr_scaler.transform(prematch_light):.2f} mw {pr_scaler.transform(win_team):.2f} md {gt_scaler.transform(int(dur)):.3f} tk {tk_scaler.transform(int(tk)):.3f} > """

            events = "> ".join(v for k, v in sorted_events)
            events += "<|endoftext|>"
            maps[str(cur_map_id)] = prematch + events
        except:
            pass

    return maps


if not os.path.exists(MAPS_OUT):
    maps = extract_dota2_v1()
    with open(MAPS_OUT, "w") as outfile:
        json.dump(maps, outfile)


# _________________________ TOKENIZE  _________________________________________


def is_number(s):
    try:
        return math.isfinite(float(s))
    except ValueError:
        return False


with open(
    "/home/user/ODDIN/llm/MjolnirMind/nanoGPT/data/dota2/maps_v7.json", "r"
) as readfile:
    data_v1 = json.load(readfile)

# with open(
#     "/home/user/ODDIN/llm/MjolnirMind/nanoGPT/data/dota2/maps_opendota.json", "r"
# ) as readfile:
#     data_opendota = json.load(readfile)

data = data_v1
# data = data_v1 | dict(list(data_opendota.items())[-len(data_v1) :])

# with open(
#     "/home/user/ODDIN/llm/MjolnirMind/nanoGPT/data/dota2/maps_v3.json", "w"
# ) as outfile:
#     json.dump(data, outfile)


with open(f"{OUT_DATA_PATH}/hero_ids.json", "r") as readfile:
    hero_ids = json.load(readfile)

data = list(data.values())

hero_vocab = [v for k, v in hero_ids.items()]
side_vocab = ["light", "dark", "team"]

vocab = [
    x
    for x in sorted(list(set().union(*[x.split(" ") for x in data])))
    if x != "<|endoftext|>"
    and x not in hero_vocab
    and not is_number(x)
    and x not in side_vocab
]

vocab = ["<|pad|>", "<|endoftext|>", "<|num|>"] + side_vocab + hero_vocab + vocab
vocab_size = len(vocab)
enc = {ch: i for i, ch in enumerate(vocab)}
dec = {i: ch for i, ch in enumerate(vocab)}

meta = {
    "vocab_size": vocab_size,
    "enc": enc,
    "dec": dec,
}
with open(f"{OUT_DATA_PATH}/meta_num.pkl", "wb") as f:
    pickle.dump(meta, f)
