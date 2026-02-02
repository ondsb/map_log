import os

import h5py
import pandas as pd
from tqdm import tqdm


def print_key_cols():
    temp_hdf = h5py.File(f"{DATA_PATH}/dota2_all.h5", "r")
    temp_hdf.keys()

    for k in temp_hdf.keys():
        z = pd.read_hdf(f"{DATA_PATH}/hdf_files/330917_425662_7151729097.h5", k)
        print(k)
        print(z.columns)
        print()


def get_hero_ids():
    import requests
    import json

    def get_heroes(api_key):
        url = f"https://api.steampowered.com/IEconDOTA2_570/GetHeroes/v0001/?key={api_key}&language=en_us"
        response = requests.get(url)
        data = json.loads(response.text)
        return data["result"]["heroes"]

    api_key = "B211FE49B2C04912798FD33E75857D1E"
    heroes = get_heroes(api_key)

    heroes = {x["id"]: x["name"].split("hero_")[1] for x in heroes}

    with open(f"{OUT_DATA_PATH}/hero_ids.json", "w") as outfile:
        json.dump(heroes, outfile)


def merge_hdfs():
    files = os.listdir(f"{DATA_PATH}/merged_files")

    for key in [
        "snapshots",
        "status",
        "player_kills",
        "team_stats",
        "building_kills",
        "neutral_kills",
        "pick_and_ban",
    ]:
        print(key)
        big = []
        for file in tqdm(files):
            try:
                big.append(pd.read_hdf(f"{DATA_PATH}/merged_files/{file}", key))
            except:
                continue
        big = pd.concat(big).reset_index(drop=True)
        print(big.shape[0])
        big.to_hdf(f"{DATA_PATH}/dota2_all.hdf", key, format="table", data_columns=True)


def reformat_hdfs():
    files = os.listdir(f"{DATA_PATH}/merged_files")

    for key in [
        "snapshots",
        "status",
        "player_kills",
        "team_stats",
        "building_kills",
        "neutral_kills",
        "pick_and_ban",
    ][::-1]:
        for file in tqdm(files):
            try:
                df = pd.read_hdf(f"{DATA_PATH}/merged_files/{file}", key)

                if key == "status":
                    if "oddin_home_team_parent_id" in df.columns:
                        df[
                            ["oddin_home_team_parent_id", "oddin_away_team_parent_id"]
                        ] = df[
                            ["oddin_home_team_parent_id", "oddin_away_team_parent_id"]
                        ].astype(
                            float
                        )
                    df["oddin_match_id"] = df["oddin_match_id"].astype(float)
                    df["valve_map_id"] = df["valve_map_id"].astype(float)
                    df["duration"] = df["duration"].astype(float)
                    if "prematch_only" in df.columns:
                        df["prematch_only"] = df["prematch_only"].astype(float)

                if key in ["player_kills", "team_stats"]:
                    if "first_barracks_valve_team_id" in df.columns:
                        df["first_barracks_valve_team_id"] = df[
                            "first_barracks_valve_team_id"
                        ].astype(float)

                if key == "team_stats":
                    if "first_roshan_kill_valve_team_id" in df.columns:
                        df["first_roshan_kill_valve_team_id"] = df[
                            "first_roshan_kill_valve_team_id"
                        ].astype(float)
                    if "first_tower_valve_team_id" in df.columns:
                        df["first_tower_valve_team_id"] = df[
                            "first_tower_valve_team_id"
                        ].astype(float)

                df.to_hdf(
                    f"{DATA_PATH}/merged_files_/{file}.hdf",
                    key,
                    format="table",
                    data_columns=True,
                )
            except Exception as e:
                print(e)


def process_files():
    files = sorted(os.listdir(DATA_PATH))

    file_path = os.path.join(DATA_PATH, files[0])
    keys = h5py.File(file_path, "r").keys()

    for key in keys:
        print(key)
        big = []
        for file in tqdm(files):
            if file.endswith(".hdf"):  # Check if the file has the .h5 extension
                file_path = os.path.join(DATA_PATH, file)
                try:
                    big.append(pd.read_hdf(file_path, key))
                except Exception as e:
                    print(f"Error processing {file}: {e}")
        pd.concat(big).reset_index(drop=True).to_hdf(
            f"{DATA_PATH}/../dota2_all.h5", key
        )


def prefinal_process_files():
    source = "/home/user/ODDIN/data/dota2/merged_zfinal/dota2_2022_2023.h5"
    store_path = "/home/user/ODDIN/data/dota2/merged_zfinal/merged_all.h5"

    with pd.HDFStore(source, mode="r") as source_store:
        keys = source_store.keys()

    for key in keys:
        print(key)
        key = key[1:]
        # if key == "snapshots":
        #     continue

        cur = pd.read_hdf(source, key)

        if key == "status":
            cur["prematch_home"] = cur["prematch_home"].astype(float)
            if "map_id" in cur.columns:
                cur = cur.drop("map_id", axis=1)

        cur.to_hdf(store_path, key, format="table", data_columns=True)
        # todo 20mins+


def final_process_files():
    store_path = "/home/user/ODDIN/data/dota2/merged_zfinal/merged_all.h5"
    source = "/home/user/ODDIN/data/dota2/merged_zfinal/dota2_2020_2021.h5"

    with pd.HDFStore(source, mode="r") as source_store:
        for key in source_store.keys():
            print(key)
            key = key[1:]

            # if key == "snapshots":
            #     continue

            with pd.HDFStore(store_path, mode="a") as store:
                data = source_store[key]
                if key == "status":
                    data["trader_total_towers"] = data["trader_total_towers"].astype(
                        float
                    )
                    data["oddin_tournament_name"] = data["oddin_tournament_name"].str[
                        :60
                    ]
                    data["prematch_home"] = data["prematch_home"].astype(float)
                store.append(key, data, format="table", data_columns=True)


DATA_PATH = "/home/user/ODDIN/data/dota2/merged_files_"
OUT_DATA_PATH = "/nanoGPT/data/dota2"
MAPS_OUT = "/home/user/ODDIN/llm/MjolnirMind/nanoGPT/data/dota2/maps_v6.json"

run = False
if run:
    process_files()
    prefinal_process_files()
    final_process_files()
