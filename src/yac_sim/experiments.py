
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .config import SimConfig
from .sim_single import simulate, monte_carlo
from .utils import mean_ci95, knee_point

def apply_ieee_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "grid.linewidth": 0.4,
    })

def savefig(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{name}.png", dpi=300, bbox_inches="tight")

def _offset_x(x: np.ndarray, frac: float) -> np.ndarray:
    span = float(np.max(x) - np.min(x)) if x.size else 0.0
    if span == 0.0:
        span = max(float(np.max(x)) if x.size else 0.0, 1.0)
    return x + frac * span

def _prep_tradeoff_data(et_res, per_res, rand_res, *, y_transform: str | None = None):
    """Prepare (x, y, y_ci) curves for plotting.

    y_transform:
      - None: use J in linear scale.
      - "log10": use log10(J) and compute CI in log domain (more robust for heavy-tail).
    """
    def summarize(res_list, xkey="N_deliv"):
        xs, ys, yci = [], [], []
        for r in res_list:
            x_m, _ = mean_ci95(r[xkey])
            if y_transform == "log10":
                Jv = np.maximum(np.asarray(r["J"], dtype=float), 1e-12)
                y = np.log10(Jv)
                y_m, y_hw = mean_ci95(y)
            else:
                y_m, y_hw = mean_ci95(r["J"])
            xs.append(x_m); ys.append(y_m); yci.append(y_hw)
        xs = np.array(xs); ys = np.array(ys); yci = np.array(yci)
        order = np.argsort(xs)
        return xs[order], ys[order], yci[order], order

    x_et, y_et, ci_et, _ = summarize(et_res)
    x_per, y_per, ci_per, _ = summarize(per_res)
    x_rd, y_rd, ci_rd, _ = summarize(rand_res)
    return (x_et, y_et, ci_et), (x_per, y_per, ci_per), (x_rd, y_rd, ci_rd)


def figure_A_pareto(cfg: SimConfig, outdir: Path) -> tuple[float, float]:
    """Figure A: Pareto with multiple baselines + CI + knee. Returns (delta_knee, budget_knee_packets)."""
    apply_ieee_style()

    # Sweeps
    deltas = np.linspace(0.05, 1.0, 14).tolist()
    periods = [1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 40]
    ps = np.linspace(0.02, 0.45, 12).tolist()

    et_res = monte_carlo(cfg, "ET", deltas=deltas)
    per_res = monte_carlo(cfg, "PER", periods=periods)
    rd_res = monte_carlo(cfg, "RAND", random_ps=ps)

    (x_et, y_et, ci_et), (x_per, y_per, ci_per), (x_rd, y_rd, ci_rd) = _prep_tradeoff_data(
        et_res, per_res, rd_res, y_transform="log10"
    )

    # knee on ET curve in (packets, cost)
    knee_idx = knee_point(x_et, y_et)
    # Map knee point back to corresponding delta in ET sweep (using the same sorting as x_et).
    et_x_means = np.asarray([mean_ci95(r['N_deliv'])[0] for r in et_res], dtype=float)
    et_order = np.argsort(et_x_means)
    et_params_sorted = np.asarray([r['param'] for r in et_res], dtype=float)[et_order]
    delta_knee = float(et_params_sorted[knee_idx])
    budget_knee = float(x_et[knee_idx])

    fig = plt.figure(figsize=(3.5, 2.4))
    ax = fig.add_subplot(111)
    # Slight x-offsets separate overlapping curves without altering ordering.
    x_per_plot = _offset_x(x_per, -0.012)
    x_rd_plot = _offset_x(x_rd, 0.012)
    ax.errorbar(
        x_et, y_et, yerr=ci_et, fmt="o-",
        capsize=2, elinewidth=0.7, alpha=0.9,
        markerfacecolor="white", markeredgewidth=0.7,
        label=r"ET (sweep $\delta$)", zorder=3,
    )
    ax.errorbar(
        x_per_plot, y_per, yerr=ci_per, fmt="s--",
        capsize=2, elinewidth=0.7, alpha=0.9,
        markerfacecolor="white", markeredgewidth=0.7,
        label="PER (sweep $M$)", zorder=2,
    )
    ax.errorbar(
        x_rd_plot, y_rd, yerr=ci_rd, fmt="^:",
        capsize=2, elinewidth=0.7, alpha=0.9,
        markerfacecolor="white", markeredgewidth=0.7,
        label="RAND (sweep $p$)", zorder=1,
    )
    ax.plot([budget_knee], [y_et[knee_idx]], marker="D", linestyle="None", markersize=5, label="ET knee")

    ax.set_xlabel("Delivered communication (bits)")
    ax.set_ylabel(r"$\log_{10} J$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True, borderpad=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.margins(x=0.04)

    savefig(fig, outdir, "fig_A_pareto_tradeoff")
    plt.close(fig)
    return delta_knee, budget_knee

def figure_B_time_response(cfg: SimConfig, outdir: Path, delta_knee: float) -> None:
    """Figure B: time-domain visualization of the grow-and-reset / contraction mechanism.

    - Top: prediction error norm ||\tilde x_k|| with vertical markers indicating delivered updates.
    - Bottom: cumulative delivered uplink packets (staircase).
    Inset: innovation ||y_k - C xhat^-_k|| and the threshold δ (triggering quantity).
    """
    apply_ieee_style()

    cfg1 = SimConfig(**cfg.__dict__)
    cfg1.delta = float(delta_knee)

    rng = np.random.default_rng(int(cfg.seed) + 123)  # reproducible representative run
    out = simulate(cfg1, "ET", rng=rng)

    e_tilde = out["tilde_x_norm"]
    innov = out["innovation_norm"]
    tx = out["tx_deliv"]
    cum_pkts = np.cumsum(tx)

    t = np.arange(cfg1.T_steps) * cfg1.Ts

    fig = plt.figure(figsize=(3.5, 2.6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)

    ax1.plot(t, e_tilde, label=r"$\|\tilde x_k\|_2$")
    idx = np.where(tx > 0)[0]
    if idx.size > 0:
        ax1.vlines(t[idx], ymin=0, ymax=np.minimum(e_tilde[idx], np.max(e_tilde)), linewidth=0.6, alpha=0.6)

    ax1.set_ylabel(r"$\|\tilde x_k\|_2$")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", frameon=True, borderpad=0.3)

    ax2.step(t, cum_pkts, where="post", label="Cumulative packets")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Packets")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", frameon=True, borderpad=0.3)

    inset = inset_axes(ax1, width="38%", height="38%", loc="upper right", borderpad=0.8)
    inset.plot(t, innov, linewidth=0.9)
    inset.axhline(cfg1.delta, linestyle="--", linewidth=0.8)
    inset.set_title("innovation", fontsize=8)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_in_layout(False)

    fig.subplots_adjust(hspace=0.35, left=0.14, right=0.98, bottom=0.12, top=0.95)
    savefig(fig, outdir, "fig_B_time_response")

def figure_C_budget_curves(cfg: SimConfig, outdir: Path) -> None:
    """Figure C: relative cost gap vs ET under matched communication budgets."""
    apply_ieee_style()

    deltas = np.linspace(0.05, 1.0, 16).tolist()
    periods = [1,2,3,4,5,6,8,10,12,15,20,25,30,40]
    ps = np.linspace(0.02, 0.45, 14).tolist()

    et_res = monte_carlo(cfg, "ET", deltas=deltas)
    per_res = monte_carlo(cfg, "PER", periods=periods)
    rd_res = monte_carlo(cfg, "RAND", random_ps=ps)

    # Summarize curves using *median* (robust to heavy-tail runs).
    def summarize_median(res_list):
        xb, yb = [], []
        for r in res_list:
            xb.append(np.median(r["N_deliv"]))
            yb.append(np.median(r["J"]))
        xb = np.asarray(xb, dtype=float)
        yb = np.asarray(yb, dtype=float)
        order = np.argsort(xb)
        return xb[order], yb[order]

    x_et, y_et = summarize_median(et_res)
    x_per, y_per = summarize_median(per_res)
    x_rd, y_rd = summarize_median(rd_res)

    # Compare PER/RAND to ET at matched delivered-communication budgets.
    budgets = x_et
    per_best = np.array([_best_under_budget(x_per, y_per, b, tol=0.06) for b in budgets], dtype=float)
    rd_best = np.array([_best_under_budget(x_rd, y_rd, b, tol=0.06) for b in budgets], dtype=float)

    # Plot in log10 ratio for readability (0=equal, 1=10x worse, 2=100x worse, ...)
    ratio_per = np.log10(np.maximum(per_best / np.maximum(y_et, 1e-12), 1e-12))
    ratio_rd = np.log10(np.maximum(rd_best / np.maximum(y_et, 1e-12), 1e-12))
    # Normalize budget by full-communication periodic baseline (M=1 -> max bits).
    full_bits = float(np.max(x_per)) if x_per.size else float(np.max(x_et))
    x_norm = budgets / full_bits if full_bits > 0 else budgets

    fig = plt.figure(figsize=(3.5, 2.2))
    ax = fig.add_subplot(111)
    ax.axhline(0.0, linestyle="--", linewidth=0.9, color="0.35", label="ET baseline")
    ax.plot(
        x_norm, ratio_per, "s--",
        markerfacecolor="white", markeredgewidth=0.7,
        label="PER (best@budget)", zorder=2,
    )
    ax.plot(
        x_norm, ratio_rd, "^:",
        markerfacecolor="white", markeredgewidth=0.7,
        label="RAND (best@budget)", zorder=1,
    )
    ax.set_xlabel("Delivered communication (normalized)")
    ax.set_ylabel(r"$\log_{10}(J_{\mathrm{baseline}}/J_{\mathrm{ET}})$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True, borderpad=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.margins(x=0.04)

    savefig(fig, outdir, "fig_C_budget_curves")
    plt.close(fig)

def _best_under_budget(curve_bits: np.ndarray, curve_J: np.ndarray, budget: float, tol: float = 0.08) -> float:
    """Pick best J among points whose bits within +/- tol*budget."""
    if budget <= 0:
        return float(curve_J.min())
    lo = budget * (1 - tol)
    hi = budget * (1 + tol)
    mask = (curve_bits >= lo) & (curve_bits <= hi)
    if np.any(mask):
        return float(curve_J[mask].min())
    # fallback: nearest neighbor
    idx = int(np.argmin(np.abs(curve_bits - budget)))
    return float(curve_J[idx])

def figure_D_robustness_panel(cfg: SimConfig, outdir: Path, budget_packets: float) -> None:
    """Figure D: 2x2 robustness panel sweeping impairment strengths, comparing best ET vs best PER at same delivered budget."""
    apply_ieee_style()

    # base sweeps for knobs
    deltas = np.linspace(0.05, 1.2, 18).tolist()
    periods = [1,2,3,4,5,6,8,10,12,15,20,25,30,40]

    # parameter grids
    noise_grid = [0.0, 0.02, 0.05, 0.08, 0.10]
    bits_grid  = [32, 12, 10, 8, 6]
    loss_grid  = [0.0, 0.05, 0.10, 0.20, 0.30]
    mismatch_grid = [0.0, 0.01, 0.02, 0.04, 0.06]

    def build_curves(cfgX: SimConfig):
        et = monte_carlo(cfgX, "ET", deltas=deltas)
        per = monte_carlo(cfgX, "PER", periods=periods)
        # summarize mean curves
        def summarize(res_list):
            xb, yb = [], []
            for r in res_list:
                x_m, _ = mean_ci95(r["N_deliv"])
                y_m, _ = mean_ci95(r["J"])
                xb.append(x_m); yb.append(y_m)
            xb = np.array(xb); yb = np.array(yb)
            order = np.argsort(xb)
            return xb[order], yb[order]
        xet, yet = summarize(et)
        xpe, ype = summarize(per)
        return xet, yet, xpe, ype

    # nominal for normalization
    cfg_nom = SimConfig(**cfg.__dict__)
    cfg_nom.mode = "robust"  # normalization uses same mode family as panel
    xet0, yet0, xpe0, ype0 = build_curves(cfg_nom)
    J_ref = _best_under_budget(xpe0, ype0, budget_packets)  # reference: PER at that budget

    def eval_grid(var_name: str, grid):
        et_best = []
        per_best = []
        for val in grid:
            c = SimConfig(**cfg.__dict__)
            c.mode = "robust"
            # default robust assumptions
            c.sigma_w = max(c.sigma_w, 0.02)
            if var_name == "sigma_v":
                c.sigma_v = float(val)
            elif var_name == "bits":
                c.bits_per_value = int(val)
            elif var_name == "loss":
                # set both states to same average loss for sweep clarity
                c.loss_good = float(val)
                c.loss_bad = float(val)
            elif var_name == "mismatch":
                c.mismatch_eps = float(val)
            else:
                raise ValueError(var_name)

            xet, yet, xpe, ype = build_curves(c)
            et_best.append(_best_under_budget(xet, yet, budget_packets) / J_ref)
            per_best.append(_best_under_budget(xpe, ype, budget_packets) / J_ref)
        return np.array(et_best), np.array(per_best)

    et_noise, per_noise = eval_grid("sigma_v", noise_grid)
    et_bits,  per_bits  = eval_grid("bits", bits_grid)
    et_loss,  per_loss  = eval_grid("loss", loss_grid)
    et_mis,   per_mis   = eval_grid("mismatch", mismatch_grid)

    fig = plt.figure(figsize=(3.5, 3.2))
    axs = [fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(223), fig.add_subplot(224)]

    axs[0].plot(noise_grid, et_noise, marker="o", label="ET")
    axs[0].plot(noise_grid, per_noise, marker="s", label="PER")
    axs[0].set_title("Measurement noise")
    axs[0].set_xlabel(r"$\sigma_v$")
    axs[0].set_ylabel("Cost ratio")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(bits_grid, et_bits, marker="o", label="ET")
    axs[1].plot(bits_grid, per_bits, marker="s", label="PER")
    axs[1].set_title("Quantization")
    axs[1].set_xlabel("bits/value")
    axs[1].grid(True, alpha=0.3)
    axs[1].invert_xaxis()  # fewer bits to the right in IEEE figures often confusing; invert for monotonic degradation to right? keep inverted for visual.

    axs[2].plot(loss_grid, et_loss, marker="o", label="ET")
    axs[2].plot(loss_grid, per_loss, marker="s", label="PER")
    axs[2].set_title("Packet loss")
    axs[2].set_xlabel("loss prob.")
    axs[2].set_ylabel("Cost ratio")
    axs[2].grid(True, alpha=0.3)

    axs[3].plot(mismatch_grid, et_mis, marker="o", label="ET")
    axs[3].plot(mismatch_grid, per_mis, marker="s", label="PER")
    axs[3].set_title("Model mismatch")
    axs[3].set_xlabel(r"$\epsilon$ in $A+\epsilon I$")
    axs[3].grid(True, alpha=0.3)

    # single legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(pad=0.8, rect=[0, 0, 1, 0.97])

    savefig(fig, outdir, "fig_D_robustness_panel")
    plt.close(fig)

def _apply_contrast_settings(cfg: SimConfig) -> None:
    """Apply realistic-but-hard impairments so ET vs PER/RAND separation is visible.

    Key idea: keep the plant stable (double integrator), but make *estimation* and
    *delivery* the bottlenecks via bursty packet loss, quantization, and model mismatch.
    """
    cfg.mode = "robust"

    # process/measurement noise: moderate (avoid pure-noise dominance)
    cfg.sigma_w = max(cfg.sigma_w, 0.08)
    cfg.sigma_v = max(cfg.sigma_v, 0.05)

    # quantization: make it matter (and keep range tighter so resolution is challenged)
    cfg.bits_per_value = min(cfg.bits_per_value, 8)
    cfg.q_min = -12.0
    cfg.q_max = 12.0

    # stronger bursty channel (Gilbert–Elliott)
    cfg.p_good_to_bad = max(cfg.p_good_to_bad, 0.03)
    cfg.p_bad_to_good = min(cfg.p_bad_to_good, 0.08)
    cfg.loss_good = max(cfg.loss_good, 0.02)
    cfg.loss_bad = max(cfg.loss_bad, 0.45)

    # stronger model mismatch (main driver of ET advantage without changing plant stability)
    cfg.mismatch_eps = max(cfg.mismatch_eps, 0.08)


def run_all(outdir: str = "result", mc_runs: int | None = None, t_steps: int | None = None, fast: bool = False) -> None:
    outdir = Path(outdir)
    # 1) figure A,B,C use a realistic-impairment setting to accentuate gaps
    cfg_main = SimConfig()
    if fast:
        if mc_runs is None:
            mc_runs = 6
        if t_steps is None:
            t_steps = 200
    if mc_runs is not None:
        cfg_main.mc_runs = int(mc_runs)
    if t_steps is not None:
        cfg_main.T_steps = int(t_steps)
    if mc_runs is not None or t_steps is not None:
        print(f"[yac] running with mc_runs={cfg_main.mc_runs}, T_steps={cfg_main.T_steps}")
    _apply_contrast_settings(cfg_main)

    delta_knee, budget_knee = figure_A_pareto(cfg_main, outdir)
    figure_B_time_response(cfg_main, outdir, delta_knee)
    figure_C_budget_curves(cfg_main, outdir)

    # 2) figure D robustness uses robust mode and budget picked from theory knee as a concrete operating point
    cfg_robust = SimConfig(**cfg_main.__dict__)
    cfg_robust.mode = "robust"
    cfg_robust.sigma_v = max(cfg_robust.sigma_v, 0.03)
    cfg_robust.bits_per_value = min(cfg_robust.bits_per_value, 10)
    cfg_robust.loss_good = max(cfg_robust.loss_good, 0.05)
    cfg_robust.loss_bad = max(cfg_robust.loss_bad, 0.20)
    cfg_robust.mismatch_eps = max(cfg_robust.mismatch_eps, 0.02)
    figure_D_robustness_panel(cfg_robust, outdir, budget_packets=budget_knee)
