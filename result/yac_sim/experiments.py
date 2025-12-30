from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import SimConfig
from .sim_multi import simulate_two_uav
from .sim_single import simulate_single_uav


def monte_carlo_single(cfg: SimConfig, policy: str, runs: int = 30, **kwargs):
    stats = []
    for i in range(runs):
        c = SimConfig(**{**cfg.__dict__, "seed": cfg.seed + i})
        _, s = simulate_single_uav(c, policy, **kwargs)
        stats.append(s)
    df = pd.DataFrame(stats)
    return df, {
        "rms_mean": df["rms_err"].mean(),
        "rms_std": df["rms_err"].std(),
        "tx_rate_mean": df["tx_rate"].mean(),
        "bits_mean": df["bits_used"].mean(),
        "energy_mean": df["energy"].mean(),
        "fail_rate": df["failed"].mean(),
        "avg_qerr": df["avg_qerr"].mean(),
    }


def monte_carlo_two(cfg: SimConfig, base_policy: str, runs: int = 30):
    stats = []
    for i in range(runs):
        c = SimConfig(**{**cfg.__dict__, "seed": cfg.seed + i})
        _, s = simulate_two_uav(c, base_policy=base_policy)
        stats.append(s)
    df = pd.DataFrame(stats)
    return df, {
        "rms_avg_mean": df["rms_err_avg"].mean(),
        "bits_mean": df["bits_used"].mean(),
        "tx_meas_mean": df["N_tx_meas"].mean(),
        "tx_share_mean": df["N_tx_share"].mean(),
        "fail_rate": df["failed"].mean(),
    }


def run_experiments(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    base = SimConfig()

    bits_list = [4, 6, 8, 10, 12]
    rows = []
    for b in bits_list:
        cfg = SimConfig(**{**base.__dict__, "bits_per_value": b, "delta": 0.5})
        _, summ = monte_carlo_single(cfg, "event", runs=30)
        rows.append({"bits_per_value": b, **summ})
    df_q = pd.DataFrame(rows)
    df_q.to_csv(output_dir / "exp_quant_tradeoff.csv", index=False)

    plt.figure()
    plt.plot(df_q["bits_mean"], df_q["rms_mean"], marker="o")
    plt.xlabel("Average bits used (total)")
    plt.ylabel("RMS tracking error (m)")
    plt.title("Rate--distortion--control trade-off (quantization)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_quant_tradeoff.png", dpi=200)

    p_bad_list = [0.3, 0.5, 0.7]
    burst_list = [(0.01, 0.2), (0.02, 0.1), (0.05, 0.05)]
    rob_rows = []
    for p_bad in p_bad_list:
        for (p_g2b, p_b2g) in burst_list:
            cfg = SimConfig(
                **{
                    **base.__dict__,
                    "p_loss_bad": p_bad,
                    "p_g2b": p_g2b,
                    "p_b2g": p_b2g,
                    "delta": 0.5,
                    "bits_per_value": 8,
                }
            )
            _, summ = monte_carlo_single(cfg, "event", runs=30)
            rob_rows.append({"p_loss_bad": p_bad, "p_g2b": p_g2b, "p_b2g": p_b2g, **summ})
    df_rob = pd.DataFrame(rob_rows)
    df_rob.to_csv(output_dir / "exp_markov_robustness.csv", index=False)

    plt.figure()
    sub = df_rob[df_rob["p_loss_bad"] == 0.5].copy()
    sub["burst_len_proxy"] = 1.0 / (sub["p_b2g"] + 1e-12)
    for (p_g2b, p_b2g) in burst_list:
        s2 = sub[(sub["p_g2b"] == p_g2b) & (sub["p_b2g"] == p_b2g)]
        if len(s2) == 0:
            continue
        plt.scatter(s2["burst_len_proxy"], s2["rms_mean"], label=f"g2b={p_g2b}, b2g={p_b2g}")
    plt.xlabel("Bad-state burst length proxy (1/p_b2g)")
    plt.ylabel("RMS tracking error (m)")
    plt.title("Robustness under bursty Markov losses (p_bad=0.5)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig_markov_robustness.png", dpi=200)

    base2 = SimConfig(**{**base.__dict__, "multi_uav": True, "bit_budget_total": base.bit_budget_total})
    cfg_noshare = SimConfig(**{**base2.__dict__, "share_pose": False})
    _, s_noshare = monte_carlo_two(cfg_noshare, base_policy="event", runs=20)

    cfg_share = SimConfig(
        **{**base2.__dict__, "share_pose": True, "share_policy": "event", "share_delta": 2.0}
    )
    _, s_share = monte_carlo_two(cfg_share, base_policy="event", runs=20)

    comp = pd.DataFrame(
        [
            {"setting": "2-UAV, no sharing", **s_noshare},
            {"setting": "2-UAV, event-triggered pose sharing", **s_share},
        ]
    )
    comp.to_csv(output_dir / "exp_two_uav_compare.csv", index=False)
    print("Two-UAV comparison:\n", comp)

    df_traj, _ = simulate_two_uav(cfg_share, base_policy="event")
    plt.figure()
    plt.plot(df_traj["px1_ref"], df_traj["py1_ref"], "--", label="ref UAV1")
    plt.plot(df_traj["px1"], df_traj["py1"], label="UAV1")
    plt.plot(df_traj["px2_ref"], df_traj["py2_ref"], "--", label="ref UAV2")
    plt.plot(df_traj["px2"], df_traj["py2"], label="UAV2")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Two-UAV field coverage (with sharing)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig_two_uav_trajectory.png", dpi=200)

    print(
        "Saved figures & CSVs:",
        "fig_quant_tradeoff.png, fig_markov_robustness.png, fig_two_uav_trajectory.png",
        "exp_quant_tradeoff.csv, exp_markov_robustness.csv, exp_two_uav_compare.csv",
    )
