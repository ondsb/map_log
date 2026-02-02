import dataclasses


@dataclasses.dataclass
class ScalerPars:
    tar_min: float
    tar_max: float
    obs_min: float
    obs_max: float


class Scaler:
    w: ScalerPars

    def __init__(self, scaler_pars: ScalerPars) -> None:
        self.w = scaler_pars

    def transform(self, value: float) -> float:
        return (self.w.tar_max - self.w.tar_min) * (value - self.w.obs_min) / (
            self.w.obs_max - self.w.obs_min
        ) + self.w.tar_min

    def inverse_transform(self, scaled_value: float) -> float:
        return (
            (scaled_value - self.w.tar_min)
            * (self.w.obs_max - self.w.obs_min)
            / (self.w.tar_max - self.w.tar_min)
        ) + self.w.obs_min


class Sc:
    gt = Scaler(ScalerPars(0, 5, 0, 6500))
    tk = Scaler(ScalerPars(0, 5, 0, 150))
    pk = Scaler(ScalerPars(0, 5, 0, 50))
    pr = Scaler(ScalerPars(-3, 3, 0, 1))
    kd = Scaler(ScalerPars(-3, 3, -50, 50))
