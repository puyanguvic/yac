from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimConfig:
    # -------------------------
    # Simulation timing
    # -------------------------
    Ts: float = 0.1
    T_steps: int = 2000

    # -------------------------
    # Reference / mission (lawnmower)
    # -------------------------
    field_L: float = 200.0
    field_W: float = 100.0
    lane_spacing: float = 10.0
    v_ref: float = 2.0

    # -------------------------
    # Plant / controller (double integrator + fixed LQR)
    # -------------------------
    Q: np.ndarray = field(default_factory=lambda: np.diag([10.0, 1.0, 10.0, 1.0]))
    R: np.ndarray = field(default_factory=lambda: 0.1 * np.eye(2))
    u_max: float = 8.0

    # -------------------------
    # Event-trigger threshold (paper knob)
    # -------------------------
    delta: float = 1.0

    # -------------------------
    # Measurement noise
    # -------------------------
    sigma_v: float = 0.0

    # Process disturbance (model mismatch / wind). Adds w_k to plant update.
    sigma_w: float = 0.05

    # -------------------------
    # Communication / quantization
    # -------------------------
    bits_per_value: int = 8
    bits_per_packet_overhead: int = 64
    bit_budget_total: int = 10_000_000  # total bits over the horizon

    # If True, lower bits in bad channel state (a toy "adaptive coding" option)
    adaptive_bits: bool = False

    # Gilbert–Elliott channel parameters (loss prob in good/bad; transition probs)
    p_loss_good: float = 0.02
    p_loss_bad: float = 0.30
    p_g2b: float = 0.01
    p_b2g: float = 0.10

    # -------------------------
    # Theory vs robust modes (NEW)
    # -------------------------
    # "theory": matches paper baseline (perfect measurement, ideal reset-on-receive).
    # "robust": uses noise/quantization/channel loss as configured above.
    mode: str = "robust"

    # If True, enforce perfect measurement y=x (used by mode="theory")
    ideal_measurement: bool = False
    # If True, every attempted transmission is delivered (used by mode="theory")
    ideal_comm: bool = False
    # If True, received sample is not quantized (used by mode="theory")
    ideal_quant: bool = False

    # -------------------------
    # Multi-agent options (kept for compatibility)
    # -------------------------
    base_policy: str = "event"
    base_period_M: int = 10
    base_random_q: float = 0.2

    share_pose: bool = True
    share_policy: str = "event"
    share_period_M: int = 10
    share_random_q: float = 0.2
    share_delta: float = 1.0
    share_bits_per_value: int = 8

    # -------------------------
    # Failure definition
    # -------------------------
    fail_err: float = 20.0
    fail_window: int = 50

    # -------------------------
    # RNG seed
    # -------------------------
    seed: int = 0

    def normalize_modes(self) -> None:
        """If mode='theory', enforce ideal switches and remove exogenous impairments."""
        if self.mode == "theory":
            self.ideal_measurement = True
            self.ideal_comm = True
            self.ideal_quant = True
            self.sigma_v = 0.0
            self.p_loss_good = 0.0
            self.p_loss_bad = 0.0
