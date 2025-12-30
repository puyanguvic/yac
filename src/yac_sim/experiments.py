from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
        "J_mean": df["J_cost"].mean(),
        "N_tx_mean": df["N_tx"].mean(),
        "N_tx_attempt_mean": df["N_tx_attempt"].mean(),
        "tx_rate_mean": df["tx_rate"].mean(),
        "tx_attempt_rate_mean": df["tx_attempt_rate"].mean(),
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

    delta_list = np.logspace(-2, 0.5, num=9)
    pareto_rows = []
    for delta in delta_list:
        cfg = SimConfig(**{**base.__dict__, "delta": delta})
        _, summ = monte_carlo_single(cfg, "event", runs=30)
        pareto_rows.append(
            {
                "delta": delta,
                "J_mean": summ["J_mean"],
                "N_tx_mean": summ["N_tx_mean"],
                "tx_rate_mean": summ["tx_rate_mean"],
                "bits_mean": summ["bits_mean"],
            }
        )
    df_pareto = pd.DataFrame(pareto_rows)
    df_pareto.to_csv(output_dir / "exp_pareto_tradeoff.csv", index=False)

    plt.figure()
    plt.plot(df_pareto["N_tx_mean"], df_pareto["J_mean"], marker="o")
    plt.xlabel("Average delivered updates")
    plt.ylabel("Quadratic cost J")
    plt.title("Performance--communication trade-off (event-triggered)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_pareto_tradeoff.png", dpi=200)

    time_delta_list = [
        float(delta_list[0]),
        float(delta_list[len(delta_list) // 2]),
        float(delta_list[-1]),
    ]
    time_rows = []
    fig, axes = plt.subplots(len(time_delta_list), 1, figsize=(8, 8), sharex=True)
    if len(time_delta_list) == 1:
        axes = [axes]
    for ax, delta in zip(axes, time_delta_list):
        cfg = SimConfig(**{**base.__dict__, "delta": delta})
        df, _ = simulate_single_uav(cfg, "event")
        df = df.copy()
        df["delta"] = delta
        time_rows.append(df[["k", "delta", "tilde_norm", "x_norm", "tx"]])

        ax.plot(df["k"], df["tilde_norm"], label="||tilde_x||")
        ax.plot(df["k"], df["x_norm"], label="||x||", linestyle="--")
        ax.axhline(delta, color="k", linestyle=":", linewidth=1.0, label="delta")
        ax.scatter(df.loc[df["tx"] == 1, "k"], [0.0] * int(df["tx"].sum()), s=8, alpha=0.5)
        ax.set_ylabel("norm")
        ax.set_title(f"time response (delta={delta:.3f})")
        ax.grid(True)
    axes[-1].set_xlabel("time step k")
    axes[0].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_time_response.png", dpi=200)

    if time_rows:
        df_time = pd.concat(time_rows, ignore_index=True)
        df_time.to_csv(output_dir / "exp_time_response.csv", index=False)

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

    budget_list = [200_000, 500_000, 1_000_000, 2_000_000]
    budget_rows = []
    for budget in budget_list:
        cfg = SimConfig(**{**base.__dict__, "bit_budget_total": budget, "delta": 0.5})
        _, summ = monte_carlo_single(cfg, "event", runs=30)
        budget_rows.append(
            {
                "bit_budget_total": budget,
                "J_mean": summ["J_mean"],
                "bits_mean": summ["bits_mean"],
                "rms_mean": summ["rms_mean"],
                "N_tx_mean": summ["N_tx_mean"],
            }
        )
    df_budget = pd.DataFrame(budget_rows)
    df_budget.to_csv(output_dir / "exp_budget_tradeoff.csv", index=False)

    plt.figure()
    plt.plot(df_budget["bits_mean"], df_budget["J_mean"], marker="o")
    plt.xlabel("Average bits used (total)")
    plt.ylabel("Quadratic cost J")
    plt.title("Budget robustness (event-triggered)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_budget_tradeoff.png", dpi=200)

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

    compare_rows = []
    compare_delta_list = [0.1, 0.5, 1.0]
    for delta in compare_delta_list:
        cfg_event = SimConfig(**{**base.__dict__, "delta": delta})
        _, summ_event = monte_carlo_single(cfg_event, "event", runs=30)

        target_updates = max(1.0, summ_event["N_tx_mean"])
        period_updates = max(1, int(round(cfg_event.T_steps / target_updates)))
        _, summ_period_updates = monte_carlo_single(
            cfg_event, "periodic", runs=30, periodic_M=period_updates
        )
        compare_rows.append(
            {
                "delta": delta,
                "match": "updates",
                "event_J": summ_event["J_mean"],
                "event_bits": summ_event["bits_mean"],
                "event_N_tx": summ_event["N_tx_mean"],
                "periodic_M": period_updates,
                "periodic_J": summ_period_updates["J_mean"],
                "periodic_bits": summ_period_updates["bits_mean"],
                "periodic_N_tx": summ_period_updates["N_tx_mean"],
            }
        )

        bits_per_packet = cfg_event.bits_per_packet_overhead + 4 * cfg_event.bits_per_value
        target_bits = max(bits_per_packet, summ_event["bits_mean"])
        target_attempts = target_bits / bits_per_packet
        period_bits = max(1, int(round(cfg_event.T_steps / max(1.0, target_attempts))))
        _, summ_period_bits = monte_carlo_single(
            cfg_event, "periodic", runs=30, periodic_M=period_bits
        )
        compare_rows.append(
            {
                "delta": delta,
                "match": "bits",
                "event_J": summ_event["J_mean"],
                "event_bits": summ_event["bits_mean"],
                "event_N_tx": summ_event["N_tx_mean"],
                "periodic_M": period_bits,
                "periodic_J": summ_period_bits["J_mean"],
                "periodic_bits": summ_period_bits["bits_mean"],
                "periodic_N_tx": summ_period_bits["N_tx_mean"],
            }
        )

    df_compare = pd.DataFrame(compare_rows)
    df_compare.to_csv(output_dir / "exp_periodic_comparison.csv", index=False)

    base2 = SimConfig(**{**base.__dict__, "multi_uav": True, "bit_budget_total": base.bit_budget_total})
    base_policies = ["event", "periodic", "random"]
    comp_rows = []
    for policy in base_policies:
        cfg_noshare = SimConfig(**{**base2.__dict__, "share_pose": False, "base_policy": policy})
        _, s_noshare = monte_carlo_two(cfg_noshare, base_policy=policy, runs=20)
        comp_rows.append({"setting": f"2-UAV, no sharing ({policy})", **s_noshare})

        cfg_share = SimConfig(
            **{
                **base2.__dict__,
                "share_pose": True,
                "share_policy": "event",
                "share_delta": 2.0,
                "base_policy": policy,
            }
        )
        _, s_share = monte_carlo_two(cfg_share, base_policy=policy, runs=20)
        comp_rows.append({"setting": f"2-UAV, event-triggered pose sharing ({policy})", **s_share})

    comp = pd.DataFrame(comp_rows)
    comp.to_csv(output_dir / "exp_two_uav_baselines.csv", index=False)
    print("Two-UAV baselines:\n", comp)

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
        "fig_pareto_tradeoff.png, fig_time_response.png, fig_quant_tradeoff.png, fig_budget_tradeoff.png, fig_markov_robustness.png, fig_two_uav_trajectory.png",
        "exp_pareto_tradeoff.csv, exp_time_response.csv, exp_quant_tradeoff.csv, exp_budget_tradeoff.csv, exp_markov_robustness.csv, exp_periodic_comparison.csv, exp_two_uav_baselines.csv",
    )
