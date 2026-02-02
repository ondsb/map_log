import dataclasses
import json
from pathlib import Path

from tqdm import tqdm

from config.train_dota2 import data_version, dataset
from model.scaler import Scaler, ScalerPars

basedir = Path(
    f"/home/user/ODDIN/llm/MjolnirMind/nanoGPT/out/{dataset}_{data_version}/matches_runs/"
)


@dataclasses.dataclass
class InferencePars:
    num_samples: int = 200
    # should be > block_size (- n_token_prompt)
    max_new_tokens: int = 2048
    temperature: float = 0.8
    top_k: int = 200


@dataclasses.dataclass
class Prompt:
    lw: str
    dw: str


class RunGenerator:
    pr_scaler: Scaler = Scaler(ScalerPars(-3, 3, 0, 1))
    prematch_light: float = 0.5
    matches_out: dict = {}

    def __init__(self, model, inf_pars: InferencePars) -> None:
        self.model = model
        self.inf_pars = inf_pars

    @property
    def lw_samples(self) -> int:
        return round(self.prematch_light * self.inf_pars.num_samples)

    @property
    def dw_samples(self) -> int:
        return self.inf_pars.num_samples - self.lw_samples

    def prepare_prompt(self, text: list[str]) -> Prompt:
        self.prematch_light = self.pr_scaler.inverse_transform(
            float(text[-1].split(" ")[1])
        )

        lw_prompt = text.copy()
        lw_prematch = lw_prompt[3].split(" ")
        lw_prematch[3] = "3.0"
        lw_prompt[3] = " ".join(lw_prematch)
        lw_prompt = " - ".join(lw_prompt) + ">"

        dw_prompt = text.copy()
        dw_prematch = dw_prompt[3].split(" ")
        dw_prematch[3] = "-3.0"
        dw_prompt[3] = " ".join(dw_prematch)
        dw_prompt = " - ".join(dw_prompt) + ">"

        return Prompt(lw_prompt, dw_prompt)

    def __call__(self, data: dict, out: bool = True) -> None:
        for i, (match_id, text) in enumerate(data.items()):
            self.matches_out = {}  # todo: export 1 json?
            print(match_id)
            prompt = self.prepare_prompt(text.split(">")[0].split(" - "))

            outs = []
            for _ in tqdm(range(self.lw_samples)):
                y = self.model.generate(
                    prompt.lw,
                    self.inf_pars.max_new_tokens,
                    self.inf_pars.temperature,
                    self.inf_pars.top_k,
                )
                outs.append(y)

            for _ in tqdm(range(self.dw_samples)):
                y = self.model.generate(
                    prompt.dw,
                    self.inf_pars.max_new_tokens,
                    self.inf_pars.temperature,
                    self.inf_pars.top_k,
                )
                outs.append(y)

            self.matches_out[match_id] = outs
            if out:
                self.save(basedir / f"run_{match_id}.json")

    def save(self, out_path: Path) -> None:
        with open(out_path, "w") as outfile:
            json.dump(self.matches_out, outfile)
