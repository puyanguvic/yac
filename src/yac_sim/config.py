from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimConfig:
    Ts: float = 0.1
    T_steps: int = 2000

    # field / reference
    field_L: float = 200.0
    field_W: float = 100.0
    lane_spacing: float = 10.0
    v_ref: float = 2.0

    # sensing noise
    sigma_v: float = 0.2

    # event trigger threshold
    delta: float = 0.5

    # LQR
    Q: np.ndarray = field(default_factory=lambda: np.diag([10.0, 1.0, 10.0, 1.0]))
    R: np.ndarray = field(default_factory=lambda: np.diag([0.1, 0.1]))
    u_max: float = 2.0

    # channel (Gilbert-Elliott)
    p_loss_good: float = 0.05
    p_loss_bad: float = 0.5
    p_g2b: float = 0.02
    p_b2g: float = 0.1

    # bit budget
    bits_per_packet_overhead: int = 64
    bits_per_value: int = 8
    bit_budget_total: int = 2_000_000
    adaptive_bits: bool = False

    # multi-UAV
    multi_uav: bool = True
    base_policy: str = "event"
    base_period_M: int = 10
    base_random_q: float = 0.2
    share_pose: bool = True
    share_policy: str = "event"
    share_period_M: int = 10
    share_random_q: float = 0.2
    share_delta: float = 1.0
    share_bits_per_value: int = 8

    # failure definition
    fail_err: float = 20.0
    fail_window: int = 50

    seed: int = 0
