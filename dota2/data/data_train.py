import json
import os
import pickle

from config.train_dota2 import data_version, dataset, meta_id

# Base path - configurable via environment variable for Docker
BASE_PATH = os.environ.get("MJOLNIR_BASE_PATH", "")

basedir = os.path.join(BASE_PATH, f"out/{dataset}_{data_version}/matches_runs/")


def get_train_val_meta(split: float = 0.98):  # todo: improve split, need seed for val
    data_dir = os.path.join(BASE_PATH, "data")
    with open(f"{data_dir}/maps_{data_version}.json", "r") as openfile:
        data = [v for k, v in json.load(openfile).items()]

    n_train = split * len(data)
    train_data = [x.split(" ") for x in data[: int(n_train)]]
    val_data = [x.split(" ") for x in data[int(n_train) :]]

    with open(os.path.join(f"{data_dir}", meta_id), "rb") as openfile:
        meta = pickle.load(openfile)

    return train_data, val_data, meta


def get_val_meta(split: float = 0.98):
    data_dir = os.path.join(BASE_PATH, "data", dataset)
    with open(os.path.join(data_dir, meta_id), "rb") as f:
        meta = pickle.load(f)

    dataset_id = f"maps_{data_version}.json"
    with open(os.path.join(data_dir, dataset_id), "rb") as f:
        matches = json.load(f)

    val = dict(list(matches.items())[int(split * len(matches)) :])

    processed_runs = [
        x.split(".")[0].split("_")[1] for x in os.listdir(basedir) if "run" in x
    ]
    val = {k: v for k, v in val.items() if k not in processed_runs}

    return val, meta
