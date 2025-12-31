
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class SimConfig:
    # timing
    Ts: float = 0.1
    T_steps: int = 400
    mc_runs: int = 30
    seed: int = 7

    # modes: "theory" (paper baseline) or "robust" (noise/quant/loss/mismatch)
    mode: str = "theory"

    # trigger / periodic
    delta: float = 0.4
    period_M: int = 10
    random_p: float = 0.1  # for random baseline

    # noise / impairment
    sigma_w: float = 0.03  # process noise (needed to make grow visible)
    sigma_v: float = 0.00  # measurement noise
    bits_per_value: int = 32

    # channel (Gilbert–Elliott)
    p_good_to_bad: float =  0.03
    p_bad_to_good: float =  0.08
    loss_good: float =  0.02
    loss_bad: float =  0.45

    # model mismatch (predictor uses A_hat = A + eps*I)
    mismatch_eps: float = 0.0

    # quantizer range
    q_min: float = -20.0
    q_max: float = 20.0

    # cost matrices (as in paper)
    Q_px: float = 10.0
    Q_vx: float = 1.0
    Q_py: float = 10.0
    Q_vy: float = 1.0
    R_u: float = 0.1

    def force_theory(self) -> None:
        self.mode = "theory"
        self.sigma_v = 0.0
        self.bits_per_value = 32
        self.loss_good = 0.0
        self.loss_bad = 0.0
        self.mismatch_eps = 0.0
