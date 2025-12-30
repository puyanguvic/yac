from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import SimConfig
from .sim_multi import simulate_two_uav
from .sim_single import simulate_single_uav


def apply_plot_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "grid.alpha": 0.3,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
        }
    )


def save_figure(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


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
        "J_std": df["J_cost"].std(),
        "N_tx_mean": df["N_tx"].mean(),
        "N_tx_std": df["N_tx"].std(),
        "N_tx_attempt_mean": df["N_tx_attempt"].mean(),
        "tx_rate_mean": df["tx_rate"].mean(),
        "tx_attempt_rate_mean": df["tx_attempt_rate"].mean(),
        "bits_mean": df["bits_used"].mean(),
        "bits_std": df["bits_used"].std(),
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
        "rms_avg_std": df["rms_err_avg"].std(),
        "bits_mean": df["bits_used"].mean(),
        "bits_std": df["bits_used"].std(),
        "tx_meas_mean": df["N_tx_meas"].mean(),
        "tx_meas_std": df["N_tx_meas"].std(),
        "tx_share_mean": df["N_tx_share"].mean(),
        "tx_share_std": df["N_tx_share"].std(),
        "fail_rate": df["failed"].mean(),
    }


def run_experiments(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_plot_style()
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
                "J_std": summ["J_std"],
                "N_tx_mean": summ["N_tx_mean"],
                "N_tx_std": summ["N_tx_std"],
                "tx_rate_mean": summ["tx_rate_mean"],
                "bits_mean": summ["bits_mean"],
            }
        )
    df_pareto = pd.DataFrame(pareto_rows)
    df_pareto.to_csv(output_dir / "exp_pareto_tradeoff.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        df_pareto["N_tx_mean"],
        df_pareto["J_mean"],
        yerr=df_pareto["J_std"],
        fmt="o-",
        capsize=3,
        label="Event-triggered",
    )
    ax.set_xlabel("Average delivered updates")
    ax.set_ylabel("Quadratic cost J")
    ax.set_title("Performance--communication trade-off")
    ax.legend()
    ax.grid(True)
    save_figure(fig, output_dir / "fig_pareto_tradeoff.png")

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

        ax.plot(df["k"], df["tilde_norm"], label="||tilde_x||", color="tab:blue")
        ax.plot(df["k"], df["x_norm"], label="||x||", linestyle="--", color="tab:orange")
        ax.axhline(delta, color="k", linestyle=":", linewidth=1.0, label="delta")
        ax.scatter(
            df.loc[df["tx"] == 1, "k"],
            [0.0] * int(df["tx"].sum()),
            s=18,
            alpha=0.6,
            marker="|",
            color="tab:green",
            label="tx",
        )
        ax.set_ylabel("norm")
        ax.set_title(f"time response (delta={delta:.3f})")
        ax.grid(True)
    axes[-1].set_xlabel("time step k")
    axes[0].legend(loc="upper right", ncol=2)
    save_figure(fig, output_dir / "fig_time_response.png")

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

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        df_q["bits_mean"],
        df_q["rms_mean"],
        yerr=df_q["rms_std"],
        fmt="o-",
        capsize=3,
        color="tab:purple",
    )
    ax.set_xlabel("Average bits used (total)")
    ax.set_ylabel("RMS tracking error (m)")
    ax.set_title("Rate--distortion--control trade-off (quantization)")
    ax.grid(True)
    save_figure(fig, output_dir / "fig_quant_tradeoff.png")

    budget_list = [200_000, 500_000, 1_000_000, 2_000_000]
    budget_rows = []
    for budget in budget_list:
        cfg = SimConfig(**{**base.__dict__, "bit_budget_total": budget, "delta": 0.5})
        _, summ = monte_carlo_single(cfg, "event", runs=30)
        budget_rows.append(
            {
                "bit_budget_total": budget,
                "J_mean": summ["J_mean"],
                "J_std": summ["J_std"],
                "bits_mean": summ["bits_mean"],
                "rms_mean": summ["rms_mean"],
                "N_tx_mean": summ["N_tx_mean"],
            }
        )
    df_budget = pd.DataFrame(budget_rows)
    df_budget.to_csv(output_dir / "exp_budget_tradeoff.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        df_budget["bits_mean"],
        df_budget["J_mean"],
        yerr=df_budget["J_std"],
        fmt="o-",
        capsize=3,
        color="tab:red",
    )
    ax.set_xlabel("Average bits used (total)")
    ax.set_ylabel("Quadratic cost J")
    ax.set_title("Budget robustness (event-triggered)")
    ax.grid(True)
    save_figure(fig, output_dir / "fig_budget_tradeoff.png")

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

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sub = df_rob[df_rob["p_loss_bad"] == 0.5].copy()
    sub["burst_len_proxy"] = 1.0 / (sub["p_b2g"] + 1e-12)
    for (p_g2b, p_b2g) in burst_list:
        s2 = sub[(sub["p_g2b"] == p_g2b) & (sub["p_b2g"] == p_b2g)]
        if len(s2) == 0:
            continue
        ax.scatter(
            s2["burst_len_proxy"],
            s2["rms_mean"],
            label=f"g2b={p_g2b}, b2g={p_b2g}",
            s=50,
        )
    ax.set_xlabel("Bad-state burst length proxy (1/p_b2g)")
    ax.set_ylabel("RMS tracking error (m)")
    ax.set_title("Robustness under bursty Markov losses (p_bad=0.5)")
    ax.grid(True)
    ax.legend()
    save_figure(fig, output_dir / "fig_markov_robustness.png")

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
                "event_J_std": summ_event["J_std"],
                "event_bits": summ_event["bits_mean"],
                "event_bits_std": summ_event["bits_std"],
                "event_N_tx": summ_event["N_tx_mean"],
                "periodic_M": period_updates,
                "periodic_J": summ_period_updates["J_mean"],
                "periodic_J_std": summ_period_updates["J_std"],
                "periodic_bits": summ_period_updates["bits_mean"],
                "periodic_bits_std": summ_period_updates["bits_std"],
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
                "event_J_std": summ_event["J_std"],
                "event_bits": summ_event["bits_mean"],
                "event_bits_std": summ_event["bits_std"],
                "event_N_tx": summ_event["N_tx_mean"],
                "periodic_M": period_bits,
                "periodic_J": summ_period_bits["J_mean"],
                "periodic_J_std": summ_period_bits["J_std"],
                "periodic_bits": summ_period_bits["bits_mean"],
                "periodic_bits_std": summ_period_bits["bits_std"],
                "periodic_N_tx": summ_period_bits["N_tx_mean"],
            }
        )

    df_compare = pd.DataFrame(compare_rows)
    df_compare.to_csv(output_dir / "exp_periodic_comparison.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True)
    for match, linestyle in [("updates", "-"), ("bits", "--")]:
        sub = df_compare[df_compare["match"] == match].sort_values("delta")
        axes[0].errorbar(
            sub["delta"],
            sub["event_J"],
            yerr=sub["event_J_std"],
            marker="o",
            linestyle=linestyle,
            capsize=3,
            label=f"Event ({match})",
        )
        axes[0].errorbar(
            sub["delta"],
            sub["periodic_J"],
            yerr=sub["periodic_J_std"],
            marker="s",
            linestyle=linestyle,
            capsize=3,
            label=f"Periodic ({match})",
        )
        axes[1].errorbar(
            sub["delta"],
            sub["event_bits"],
            yerr=sub["event_bits_std"],
            marker="o",
            linestyle=linestyle,
            capsize=3,
            label=f"Event ({match})",
        )
        axes[1].errorbar(
            sub["delta"],
            sub["periodic_bits"],
            yerr=sub["periodic_bits_std"],
            marker="s",
            linestyle=linestyle,
            capsize=3,
            label=f"Periodic ({match})",
        )
    axes[0].set_ylabel("Quadratic cost J")
    axes[1].set_ylabel("Average bits used (total)")
    axes[1].set_xlabel("Event-trigger threshold delta")
    axes[0].set_title("Event vs periodic baselines")
    axes[0].legend(ncol=2, fontsize=9)
    save_figure(fig, output_dir / "fig_periodic_comparison.png")

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
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    labels = comp["setting"].str.replace("2-UAV, ", "", regex=False)
    x = np.arange(len(comp))
    axes[0].bar(
        x,
        comp["rms_avg_mean"],
        yerr=comp["rms_avg_std"],
        capsize=3,
        color="tab:blue",
    )
    axes[0].set_ylabel("Avg RMS tracking error (m)")
    axes[0].set_title("Two-UAV baselines")
    axes[1].bar(
        x,
        comp["bits_mean"],
        yerr=comp["bits_std"],
        capsize=3,
        color="tab:orange",
    )
    axes[1].set_ylabel("Average bits used (total)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    save_figure(fig, output_dir / "fig_two_uav_baselines.png")

    df_traj, _ = simulate_two_uav(cfg_share, base_policy="event")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(df_traj["px1_ref"], df_traj["py1_ref"], "--", label="ref UAV1", color="tab:blue")
    ax.plot(df_traj["px1"], df_traj["py1"], label="UAV1", color="tab:blue", alpha=0.8)
    ax.plot(df_traj["px2_ref"], df_traj["py2_ref"], "--", label="ref UAV2", color="tab:orange")
    ax.plot(df_traj["px2"], df_traj["py2"], label="UAV2", color="tab:orange", alpha=0.8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Two-UAV field coverage (with sharing)")
    ax.grid(True)
    ax.legend()
    save_figure(fig, output_dir / "fig_two_uav_trajectory.png")

    print(
        "Saved figures & CSVs:",
        "fig_pareto_tradeoff.png, fig_time_response.png, fig_quant_tradeoff.png, fig_budget_tradeoff.png, fig_markov_robustness.png, fig_periodic_comparison.png, fig_two_uav_baselines.png, fig_two_uav_trajectory.png",
        "exp_pareto_tradeoff.csv, exp_time_response.csv, exp_quant_tradeoff.csv, exp_budget_tradeoff.csv, exp_markov_robustness.csv, exp_periodic_comparison.csv, exp_two_uav_baselines.csv",
    )
