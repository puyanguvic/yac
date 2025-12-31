from __future__ import annotations

from pathlib import Path
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import SimConfig
from .sim_single import simulate_single_uav


def apply_plot_style() -> None:
    """Professional, paper-friendly Matplotlib styling."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "lines.linewidth": 1.6,
            "axes.grid": True,
            "grid.alpha": 0.35,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save a PNG (for quick viewing) + PDF (for submission)."""
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=300, facecolor="white")
    if path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def monte_carlo_single(cfg: SimConfig, policy: str, runs: int = 30, **kwargs):
    stats = []
    for i in range(runs):
        c = SimConfig(**{**asdict(cfg), "seed": cfg.seed + i})
        _, s = simulate_single_uav(c, policy, **kwargs)
        stats.append(s)
    df = pd.DataFrame(stats)
    return df, {
        "rms_mean": float(df["rms_err"].mean()),
        "rms_std": float(df["rms_err"].std()),
        "J_mean": float(df["J_cost"].mean()),
        "J_std": float(df["J_cost"].std()),
        "N_tx_mean": float(df["N_tx"].mean()),
        "N_tx_std": float(df["N_tx"].std()),
        "tx_rate_mean": float(df["tx_rate"].mean()),
        "bits_mean": float(df["bits_used"].mean()),
        "bits_std": float(df["bits_used"].std()),
    }


def run_experiments(output_dir: Path, mode: str = "paper"):
    """Entry point.

    mode:
      - "paper": generate exactly 4 publication-ready figures for the paper
    """
    if mode != "paper":
        raise ValueError("Only mode='paper' is supported in this lightweight paper build.")
    return run_experiments_paper(output_dir)


def run_experiments_paper(output_dir: Path) -> None:
    """Generate exactly four publication-ready figures.

    Figures:
      1) fig_pareto_tradeoff
      2) fig_time_response
      3) fig_periodic_comparison
      4) fig_robustness_summary

    CSVs are saved alongside the figures for reproducibility.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for p in output_dir.iterdir():
        if p.is_file():
            p.unlink()

    apply_plot_style()

    # -----------------------
    # Baseline (paper theory)
    # -----------------------
    base = SimConfig()
    base.mode = "theory"
    # Small process noise makes prediction error grow between updates (so ET is visible).
    base.sigma_w = max(base.sigma_w, 0.02)

    # ---------------------------
    # (1) Pareto trade-off curve
    # ---------------------------
    delta_list = np.array([1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0])
    pareto_rows = []
    runs_rows = []
    for d in delta_list:
        cfg = SimConfig(**{**asdict(base), "delta": float(d)})
        df_runs, summ = monte_carlo_single(cfg, policy="event", runs=30)
        pareto_rows.append(
            {
                "delta": float(d),
                "J_mean": summ["J_mean"],
                "J_std": summ["J_std"],
                "N_tx_mean": summ["N_tx_mean"],
                "N_tx_std": summ["N_tx_std"],
            }
        )
        df_runs = df_runs.copy()
        df_runs["delta"] = float(d)
        runs_rows.append(df_runs)

    df_pareto = pd.DataFrame(pareto_rows).sort_values("delta").reset_index(drop=True)
    df_pareto.to_csv(output_dir / "exp_pareto_tradeoff.csv", index=False)
    pd.concat(runs_rows, ignore_index=True).to_csv(output_dir / "exp_pareto_tradeoff_runs.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.errorbar(
        df_pareto["N_tx_mean"],
        df_pareto["J_mean"],
        xerr=df_pareto["N_tx_std"],
        yerr=df_pareto["J_std"],
        marker="o",
        linestyle="-",
        capsize=3,
    )
    ax.set_xlabel(r"Delivered updates $N_{\mathrm{tx}}$")
    ax.set_ylabel(r"Quadratic cost $J(\delta)$")
    ax.set_title("Performance–Communication Trade-off (Event-triggered)")
    save_figure(fig, output_dir / "fig_pareto_tradeoff.png")

    # -----------------------------------------
    # (2) Time response (grow-and-reset in time)
    # -----------------------------------------
    delta_mid = float(df_pareto["delta"].iloc[len(df_pareto) // 2])
    cfg_mid = SimConfig(**{**asdict(base), "delta": delta_mid, "seed": base.seed + 7})
    df_time, _ = simulate_single_uav(cfg_mid, policy="event")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(df_time["k"], df_time["x_norm"], label=r"$\|x_k\|$")
    ax.plot(df_time["k"], df_time["tilde_pred_norm"], label=r"$\|\tilde x^{pred}_k\|$")
    ax.axhline(delta_mid, linestyle="--", linewidth=1.2, label=r"threshold $\delta$")
    ax.set_xlabel("Time step $k$")
    ax.set_ylabel("Norm")
    ax.set_title("Grow-and-Reset Mechanism (Representative Run)")
    ax.legend()
    save_figure(fig, output_dir / "fig_time_response.png")

    # ---------------------------------------------------
    # (3) ET vs Periodic under matched communication usage
    # ---------------------------------------------------
    horizon = int(base.T_steps)
    N_tx_et = float(df_pareto.loc[df_pareto["delta"] == delta_mid, "N_tx_mean"].iloc[0])
    tx_rate = max(1e-6, N_tx_et / max(1, horizon))
    M = int(max(1, round(1.0 / tx_rate)))

    _, summ_et = monte_carlo_single(cfg_mid, policy="event", runs=30)
    _, summ_per = monte_carlo_single(cfg_mid, policy="periodic", runs=30, periodic_M=M)

    df_cmp = pd.DataFrame(
        [
            {"policy": "ET", "J_mean": summ_et["J_mean"], "J_std": summ_et["J_std"], "bits_mean": summ_et["bits_mean"]},
            {"policy": f"PER (M={M})", "J_mean": summ_per["J_mean"], "J_std": summ_per["J_std"], "bits_mean": summ_per["bits_mean"]},
        ]
    )
    df_cmp.to_csv(output_dir / "exp_periodic_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(
        [0, 1],
        [summ_et["J_mean"], summ_per["J_mean"]],
        yerr=[summ_et["J_std"], summ_per["J_std"]],
        capsize=4,
    )
    ax.set_xticks([0, 1], ["ET", f"PER\n(M={M})"])
    ax.set_ylabel(r"Quadratic cost $J$")
    ax.set_title("ET vs Periodic (Matched Communication)")
    save_figure(fig, output_dir / "fig_periodic_comparison.png")

    # -----------------------------------------
    # (4) Robustness summary (noise/quant/loss)
    # -----------------------------------------
    robust_cfg = SimConfig(**asdict(base))
    robust_cfg.mode = "robust"
    robust_cfg.sigma_v = 0.05
    robust_cfg.bits_per_value = 10
    robust_cfg.p_loss_good = 0.05
    robust_cfg.p_loss_bad = 0.30
    robust_cfg.p_g2b = 0.02
    robust_cfg.p_b2g = 0.20

    deltas = np.array([0.05, 0.1, 0.2, 0.5, 1.0])
    rows = []
    for d in deltas:
        c = SimConfig(**{**asdict(robust_cfg), "delta": float(d)})
        _, s_et = monte_carlo_single(c, policy="event", runs=30)
        M2 = int(max(1, round(1.0 / max(1e-6, s_et["tx_rate_mean"])) ))
        _, s_per = monte_carlo_single(c, policy="periodic", runs=30, periodic_M=M2)
        rows.append(
            {
                "delta": float(d),
                "bits_et": s_et["bits_mean"],
                "J_et": s_et["J_mean"],
                "J_et_std": s_et["J_std"],
                "bits_per": s_per["bits_mean"],
                "J_per": s_per["J_mean"],
                "J_per_std": s_per["J_std"],
                "M_per": int(M2),
            }
        )
    df_rob = pd.DataFrame(rows).sort_values("bits_et").reset_index(drop=True)
    df_rob.to_csv(output_dir / "exp_robustness_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.errorbar(df_rob["bits_et"], df_rob["J_et"], yerr=df_rob["J_et_std"], marker="o", linestyle="-", capsize=3, label="ET")
    ax.errorbar(df_rob["bits_per"], df_rob["J_per"], yerr=df_rob["J_per_std"], marker="s", linestyle="-", capsize=3, label="PER (matched)")
    ax.set_xlabel("Total transmitted bits")
    ax.set_ylabel(r"Quadratic cost $J$")
    ax.set_title("Robustness: Noise + Quantization + Bursty Loss")
    ax.legend()
    save_figure(fig, output_dir / "fig_robustness_summary.png")

    print(
        "Saved 4 paper figures:",
        "fig_pareto_tradeoff, fig_time_response, fig_periodic_comparison, fig_robustness_summary",
    )
