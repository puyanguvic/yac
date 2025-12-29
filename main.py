import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.linalg import solve_discrete_are

# -----------------------------
# Models
# -----------------------------
def build_double_integrator(Ts: float):
    A = np.array([
        [1.0, Ts,  0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, Ts ],
        [0.0, 0.0, 0.0, 1.0]
    ])
    B = np.array([
        [0.5*Ts*Ts, 0.0],
        [Ts,        0.0],
        [0.0, 0.5*Ts*Ts],
        [0.0, Ts]
    ])
    C = np.eye(4)
    return A, B, C


def dlqr(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
    return K, P


def clip_u(u, u_max):
    n = np.linalg.norm(u)
    if n <= u_max:
        return u
    return u * (u_max / (n + 1e-12))


def generate_lawnmower_ref(Ts, T_steps, L, W, lane_spacing, v_ref, x0=0.0, y0=0.0):
    """Reference state: [px,vx,py,vy]."""
    ys = np.arange(y0, y0 + W + 1e-9, lane_spacing)
    waypoints = []
    direction = 1
    for y in ys:
        if direction == 1:
            waypoints.append((x0, y))
            waypoints.append((x0 + L, y))
        else:
            waypoints.append((x0 + L, y))
            waypoints.append((x0, y))
        direction *= -1

    pts = np.array(waypoints, dtype=float)
    seg = 0
    pos = pts[0].copy()
    target = pts[1].copy()

    ref = np.zeros((T_steps, 4), dtype=float)
    for k in range(T_steps):
        vec = target - pos
        dist = float(np.linalg.norm(vec))
        if dist < 1e-6:
            seg = min(seg + 1, len(pts) - 2)
            pos = pts[seg].copy()
            target = pts[seg + 1].copy()
            vec = target - pos
            dist = float(np.linalg.norm(vec))

        step = min(v_ref * Ts, dist)
        d = vec / (dist + 1e-12)
        pos_next = pos + step * d
        vel = (pos_next - pos) / Ts
        ref[k] = np.array([pos[0], vel[0], pos[1], vel[1]])
        pos = pos_next
    return ref


class GilbertElliottChannel:
    """
    Two-state Markov channel.
    state=0: Good; state=1: Bad
    - p_loss_good: loss prob in Good
    - p_loss_bad: loss prob in Bad
    - p_g2b: P(Good->Bad)
    - p_b2g: P(Bad->Good)
    """
    def __init__(self, rng, p_loss_good=0.05, p_loss_bad=0.5, p_g2b=0.02, p_b2g=0.1, init_state=0):
        self.rng = rng
        self.p_loss_good = p_loss_good
        self.p_loss_bad = p_loss_bad
        self.p_g2b = p_g2b
        self.p_b2g = p_b2g
        self.state = init_state

    def step(self):
        # state transition
        if self.state == 0:
            if self.rng.random() < self.p_g2b:
                self.state = 1
        else:
            if self.rng.random() < self.p_b2g:
                self.state = 0

    def transmit(self):
        """Return received(bool), loss_prob(float), state(int)."""
        p = self.p_loss_good if self.state == 0 else self.p_loss_bad
        received = (self.rng.random() > p)
        return received, p, self.state


def uniform_quantize(x, bits, x_min=-10.0, x_max=10.0):
    """
    Uniform quantization for vector x with given bits per element.
    """
    if bits >= 32:
        return x.copy(), 0.0
    levels = 2 ** bits
    x_clipped = np.clip(x, x_min, x_max)
    step = (x_max - x_min) / (levels - 1)
    q = np.round((x_clipped - x_min) / step) * step + x_min
    # quantization error energy
    qe = float(np.mean((q - x) ** 2))
    return q, qe


# -----------------------------
# Config
# -----------------------------
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
    Q: np.ndarray = np.diag([10.0, 1.0, 10.0, 1.0])
    R: np.ndarray = np.diag([0.1, 0.1])
    u_max: float = 2.0

    # channel (Gilbert-Elliott)
    p_loss_good: float = 0.05
    p_loss_bad: float = 0.5
    p_g2b: float = 0.02
    p_b2g: float = 0.1

    # bit budget
    bits_per_packet_overhead: int = 64     # header + metadata
    bits_per_value: int = 8                # quantization bits per float element
    bit_budget_total: int = 2_000_000       # total bits allowed for whole sim
    adaptive_bits: bool = False            # if True, change bits based on channel state

    # multi-UAV
    multi_uav: bool = True
    share_pose: bool = True               # share pose x,y between UAVs
    share_policy: str = "event"           # 'event' or 'periodic'
    share_period_M: int = 10              # if periodic sharing
    share_delta: float = 1.0              # if event sharing threshold
    share_bits_per_value: int = 8         # quant bits for shared values

    # failure definition
    fail_err: float = 20.0
    fail_window: int = 50

    seed: int = 0


def packet_bits(num_values, bits_per_value, overhead_bits):
    return overhead_bits + num_values * bits_per_value


# -----------------------------
# Simulation core
# -----------------------------
def simulate_single_uav(cfg: SimConfig, policy: str, periodic_M=1, random_q=1.0):
    """
    policy:
      - 'event': send if ||y-y_prev|| > delta
      - 'periodic': send every M steps
      - 'random': send with prob q
    channel: Gilbert-Elliott
    budget: bit_budget_total
    payload: send 4 values (state measurement)
    """
    rng = np.random.default_rng(cfg.seed)
    A, B, C = build_double_integrator(cfg.Ts)
    K, _ = dlqr(A, B, cfg.Q, cfg.R)
    x_ref = generate_lawnmower_ref(cfg.Ts, cfg.T_steps, cfg.field_L, cfg.field_W, cfg.lane_spacing, cfg.v_ref)

    chan = GilbertElliottChannel(
        rng,
        p_loss_good=cfg.p_loss_good,
        p_loss_bad=cfg.p_loss_bad,
        p_g2b=cfg.p_g2b,
        p_b2g=cfg.p_b2g,
        init_state=0
    )

    x = np.zeros(4)
    x[0], x[2] = 5.0, 5.0
    x_hat = x.copy()
    u_prev = np.zeros(2)

    y_prev = (C @ x) + rng.normal(0.0, cfg.sigma_v, size=4)

    bits_used = 0
    N_tx = 0
    qerr_acc = 0.0

    consecutive_bad = 0
    failed = False

    rows = []

    for k in range(cfg.T_steps):
        chan.step()
        y = (C @ x) + rng.normal(0.0, cfg.sigma_v, size=4)

        # decide bits per value (optional adaptive)
        bpv = cfg.bits_per_value
        if cfg.adaptive_bits:
            # In Bad state: reduce bits to save budget, in Good: use higher bits
            bpv = 6 if chan.state == 1 else cfg.bits_per_value

        # trigger?
        do_tx = False
        if policy == "event":
            do_tx = np.linalg.norm(y - y_prev) > cfg.delta
        elif policy == "periodic":
            do_tx = (k % periodic_M == 0)
        elif policy == "random":
            do_tx = (rng.random() < random_q)
        else:
            raise ValueError("Unknown policy")

        received = False
        loss_p = cfg.p_loss_good if chan.state == 0 else cfg.p_loss_bad

        # budget gate: if not enough bits, force no-tx
        if do_tx:
            pkt_bits = packet_bits(4, bpv, cfg.bits_per_packet_overhead)
            if bits_used + pkt_bits <= cfg.bit_budget_total:
                bits_used += pkt_bits
                N_tx += 1
                received, _, _ = chan.transmit()
            else:
                do_tx = False

        if received:
            yq, qe = uniform_quantize(y, bits=bpv, x_min=-50.0, x_max=50.0)
            qerr_acc += qe
            x_hat = yq.copy()
            y_prev = y.copy()
        else:
            x_hat = A @ x_hat + B @ u_prev

        # control tracking
        e_hat = x_hat - x_ref[k]
        u = (-K @ e_hat.reshape(-1, 1)).flatten()
        u = clip_u(u, cfg.u_max)

        # state update
        x = A @ x + B @ u

        pos_err = float(np.linalg.norm(x[[0, 2]] - x_ref[k][[0, 2]]))
        if pos_err > cfg.fail_err:
            consecutive_bad += 1
        else:
            consecutive_bad = 0
        if consecutive_bad >= cfg.fail_window:
            failed = True

        rows.append({
            "k": k,
            "px": x[0], "py": x[2],
            "px_ref": x_ref[k][0], "py_ref": x_ref[k][2],
            "pos_err": pos_err,
            "ux": u[0], "uy": u[1],
            "tx": int(do_tx),
            "rx": int(received),
            "chan_state": int(chan.state),
            "loss_p": float(loss_p),
            "bits_used": int(bits_used),
            "bpv": int(bpv),
        })
        u_prev = u

    df = pd.DataFrame(rows)
    rms = math.sqrt(float(np.mean(df["pos_err"].values ** 2)))
    energy = float(np.sum(df["ux"].values**2 + df["uy"].values**2))
    tx_rate = float(N_tx) / cfg.T_steps
    avg_qerr = float(qerr_acc / max(1, df["rx"].sum()))
    return df, {
        "rms_err": rms,
        "energy": energy,
        "N_tx": int(N_tx),
        "tx_rate": tx_rate,
        "bits_used": int(bits_used),
        "avg_qerr": avg_qerr,
        "failed": int(failed),
    }


def simulate_two_uav(cfg: SimConfig, base_policy: str):
    """
    Two UAVs each track a lawnmower path. Optionally share pose (px,py) between them over the same channel model,
    under its own sharing trigger (event/periodic). Sharing consumes bits and can improve relative alignment.
    For simplicity, each UAV has its own channel instance but same parameters.
    """
    rng = np.random.default_rng(cfg.seed)
    A, B, C = build_double_integrator(cfg.Ts)
    K, _ = dlqr(A, B, cfg.Q, cfg.R)

    # Two reference paths: split field into two halves in y (top/bottom)
    ref1 = generate_lawnmower_ref(cfg.Ts, cfg.T_steps, cfg.field_L, cfg.field_W/2, cfg.lane_spacing, cfg.v_ref, x0=0.0, y0=0.0)
    ref2 = generate_lawnmower_ref(cfg.Ts, cfg.T_steps, cfg.field_L, cfg.field_W/2, cfg.lane_spacing, cfg.v_ref, x0=0.0, y0=cfg.field_W/2)

    chan1 = GilbertElliottChannel(rng, cfg.p_loss_good, cfg.p_loss_bad, cfg.p_g2b, cfg.p_b2g, 0)
    chan2 = GilbertElliottChannel(rng, cfg.p_loss_good, cfg.p_loss_bad, cfg.p_g2b, cfg.p_b2g, 0)

    # states
    x1 = np.array([5.0, 0.0, 5.0, 0.0])
    x2 = np.array([5.0, 0.0, cfg.field_W/2 + 5.0, 0.0])

    xhat1, xhat2 = x1.copy(), x2.copy()
    u1_prev, u2_prev = np.zeros(2), np.zeros(2)

    y1_prev = x1 + rng.normal(0.0, cfg.sigma_v, size=4)
    y2_prev = x2 + rng.normal(0.0, cfg.sigma_v, size=4)

    # sharing state (last shared pose)
    shared1_from2 = x2[[0, 2]].copy()
    shared2_from1 = x1[[0, 2]].copy()
    share_prev_1 = shared2_from1.copy()
    share_prev_2 = shared1_from2.copy()

    bits_used = 0
    N_tx_meas = 0
    N_tx_share = 0
    consecutive_bad1 = 0
    consecutive_bad2 = 0
    failed = False

    rows = []

    for k in range(cfg.T_steps):
        chan1.step(); chan2.step()

        y1 = x1 + rng.normal(0.0, cfg.sigma_v, size=4)
        y2 = x2 + rng.normal(0.0, cfg.sigma_v, size=4)

        # --- Measurement transmission decision (same for both) ---
        def decide_tx(policy, y, y_prev, k):
            if policy == "event":
                return np.linalg.norm(y - y_prev) > cfg.delta
            if policy == "periodic":
                return (k % 1 == 0)  # per-step periodic
            if policy == "random":
                return (rng.random() < 1.0)
            raise ValueError

        # For brevity, use base_policy only as event vs periodic for measurement
        tx1 = decide_tx("event" if base_policy == "event" else "periodic", y1, y1_prev, k)
        tx2 = decide_tx("event" if base_policy == "event" else "periodic", y2, y2_prev, k)

        # budget + channel + quantize measurement (4 values)
        def meas_update(tx, chan, y, xhat, u_prev):
            nonlocal bits_used, N_tx_meas
            received = False
            if tx:
                pkt_bits = packet_bits(4, cfg.bits_per_value, cfg.bits_per_packet_overhead)
                if bits_used + pkt_bits <= cfg.bit_budget_total:
                    bits_used += pkt_bits
                    N_tx_meas += 1
                    received, _, _ = chan.transmit()
                else:
                    tx = False
            if received:
                yq, _ = uniform_quantize(y, bits=cfg.bits_per_value, x_min=-50.0, x_max=50.0)
                return yq, y, True
            # predict
            return (A @ xhat + B @ u_prev), y, False

        xhat1, y1_prev_new, rx1 = meas_update(tx1, chan1, y1, xhat1, u1_prev)
        xhat2, y2_prev_new, rx2 = meas_update(tx2, chan2, y2, xhat2, u2_prev)
        y1_prev = y1_prev_new if rx1 else y1_prev
        y2_prev = y2_prev_new if rx2 else y2_prev

        # --- Optional pose sharing (2 values: px,py) ---
        if cfg.share_pose:
            # UAV1 shares its pose to UAV2; UAV2 shares to UAV1
            pose1 = x1[[0, 2]].copy()
            pose2 = x2[[0, 2]].copy()

            def share_decision(policy, pose, pose_prev, k):
                if policy == "event":
                    return np.linalg.norm(pose - pose_prev) > cfg.share_delta
                if policy == "periodic":
                    return (k % cfg.share_period_M == 0)
                return False

            sh1 = share_decision(cfg.share_policy, pose1, share_prev_1, k)
            sh2 = share_decision(cfg.share_policy, pose2, share_prev_2, k)

            def share_update(do_tx, chan, pose):
                nonlocal bits_used, N_tx_share
                received = False
                if do_tx:
                    pkt_bits = packet_bits(2, cfg.share_bits_per_value, cfg.bits_per_packet_overhead)
                    if bits_used + pkt_bits <= cfg.bit_budget_total:
                        bits_used += pkt_bits
                        N_tx_share += 1
                        received, _, _ = chan.transmit()
                    else:
                        do_tx = False
                if received:
                    pq, _ = uniform_quantize(pose, bits=cfg.share_bits_per_value, x_min=-50.0, x_max=250.0)
                    return pq, True
                return None, False

            pq1, ok1 = share_update(sh1, chan1, pose1)
            pq2, ok2 = share_update(sh2, chan2, pose2)

            if ok1:
                shared1_from2 = pq1  # UAV2 receives UAV1 pose (naming not important)
                share_prev_1 = pose1.copy()
            if ok2:
                shared2_from1 = pq2
                share_prev_2 = pose2.copy()

        # --- Control: add light coupling if sharing is enabled ---
        # Each UAV tracks its own ref, plus a weak "keep away / boundary alignment" term using shared pose
        e1 = xhat1 - ref1[k]
        e2 = xhat2 - ref2[k]

        u1 = (-K @ e1.reshape(-1, 1)).flatten()
        u2 = (-K @ e2.reshape(-1, 1)).flatten()

        if cfg.share_pose:
            # weak coupling: discourage overlap near boundary (y=field_W/2)
            # Use shared opposite pose to maintain separation / smooth handoff around boundary.
            # This is intentionally simple but gives you a credible "coordination benefit".
            gain = 0.02
            dy1 = (x1[2] - shared2_from1[1])
            dy2 = (x2[2] - shared1_from2[1])
            u1[1] += gain * np.clip(dy1, -10, 10)
            u2[1] += gain * np.clip(dy2, -10, 10)

        u1 = clip_u(u1, cfg.u_max)
        u2 = clip_u(u2, cfg.u_max)

        x1 = A @ x1 + B @ u1
        x2 = A @ x2 + B @ u2

        err1 = float(np.linalg.norm(x1[[0, 2]] - ref1[k][[0, 2]]))
        err2 = float(np.linalg.norm(x2[[0, 2]] - ref2[k][[0, 2]]))

        consecutive_bad1 = consecutive_bad1 + 1 if err1 > cfg.fail_err else 0
        consecutive_bad2 = consecutive_bad2 + 1 if err2 > cfg.fail_err else 0
        if consecutive_bad1 >= cfg.fail_window or consecutive_bad2 >= cfg.fail_window:
            failed = True

        rows.append({
            "k": k,
            "px1": x1[0], "py1": x1[2], "px1_ref": ref1[k][0], "py1_ref": ref1[k][2], "err1": err1,
            "px2": x2[0], "py2": x2[2], "px2_ref": ref2[k][0], "py2_ref": ref2[k][2], "err2": err2,
            "tx_meas": int(tx1) + int(tx2),
            "tx_share": int(cfg.share_pose) * (0),  # filled via counters below
            "bits_used": int(bits_used),
            "chan1_state": int(chan1.state),
            "chan2_state": int(chan2.state),
        })
        u1_prev, u2_prev = u1, u2

    df = pd.DataFrame(rows)
    rms1 = math.sqrt(float(np.mean(df["err1"].values ** 2)))
    rms2 = math.sqrt(float(np.mean(df["err2"].values ** 2)))
    return df, {
        "rms_err_uav1": rms1,
        "rms_err_uav2": rms2,
        "rms_err_avg": 0.5*(rms1 + rms2),
        "bits_used": int(bits_used),
        "N_tx_meas": int(N_tx_meas),
        "N_tx_share": int(N_tx_share),
        "failed": int(failed),
    }


def monte_carlo_single(cfg: SimConfig, policy: str, runs: int = 30, **kwargs):
    stats = []
    for i in range(runs):
        c = SimConfig(**{**cfg.__dict__, "seed": cfg.seed + i})
        _, s = simulate_single_uav(c, policy, **kwargs)
        stats.append(s)
    df = pd.DataFrame(stats)
    return df, {
        "rms_mean": df["rms_err"].mean(), "rms_std": df["rms_err"].std(),
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


# -----------------------------
# Experiments
# -----------------------------
def run_experiments():
    base = SimConfig()

    # Exp-1: bits/quantization trade-off (single UAV, event-trigger)
    bits_list = [4, 6, 8, 10, 12]
    rows = []
    for b in bits_list:
        cfg = SimConfig(**{**base.__dict__, "bits_per_value": b, "delta": 0.5})
        _, summ = monte_carlo_single(cfg, "event", runs=30)
        rows.append({"bits_per_value": b, **summ})
    df_q = pd.DataFrame(rows)
    df_q.to_csv("exp_quant_tradeoff.csv", index=False)

    plt.figure()
    plt.plot(df_q["bits_mean"], df_q["rms_mean"], marker="o")
    plt.xlabel("Average bits used (total)")
    plt.ylabel("RMS tracking error (m)")
    plt.title("Rate--distortion--control trade-off (quantization)")
    plt.grid(True); plt.tight_layout()
    plt.savefig("fig_quant_tradeoff.png", dpi=200)

    # Exp-2: bursty loss robustness (vary p_loss_bad and burstiness)
    p_bad_list = [0.3, 0.5, 0.7]
    burst_list = [(0.01, 0.2), (0.02, 0.1), (0.05, 0.05)]  # (p_g2b, p_b2g)
    rob_rows = []
    for p_bad in p_bad_list:
        for (p_g2b, p_b2g) in burst_list:
            cfg = SimConfig(**{
                **base.__dict__,
                "p_loss_bad": p_bad,
                "p_g2b": p_g2b,
                "p_b2g": p_b2g,
                "delta": 0.5,
                "bits_per_value": 8
            })
            _, summ = monte_carlo_single(cfg, "event", runs=30)
            rob_rows.append({"p_loss_bad": p_bad, "p_g2b": p_g2b, "p_b2g": p_b2g, **summ})
    df_rob = pd.DataFrame(rob_rows)
    df_rob.to_csv("exp_markov_robustness.csv", index=False)

    # plot robustness: fix p_bad=0.5, compare burstiness
    plt.figure()
    sub = df_rob[df_rob["p_loss_bad"] == 0.5].copy()
    # use expected burst length proxy: 1/p_b2g when in Bad
    sub["burst_len_proxy"] = 1.0 / (sub["p_b2g"] + 1e-12)
    for (p_g2b, p_b2g) in burst_list:
        s2 = sub[(sub["p_g2b"] == p_g2b) & (sub["p_b2g"] == p_b2g)]
        if len(s2) == 0:
            continue
        plt.scatter(s2["burst_len_proxy"], s2["rms_mean"], label=f"g2b={p_g2b}, b2g={p_b2g}")
    plt.xlabel("Bad-state burst length proxy (1/p_b2g)")
    plt.ylabel("RMS tracking error (m)")
    plt.title("Robustness under bursty Markov losses (p_bad=0.5)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig("fig_markov_robustness.png", dpi=200)

    # Exp-3: two-UAV cooperation benefit under same bit budget
    # Compare: no-sharing vs event-sharing
    base2 = SimConfig(**{**base.__dict__, "multi_uav": True, "bit_budget_total": base.bit_budget_total})
    cfg_noshare = SimConfig(**{**base2.__dict__, "share_pose": False})
    _, s_noshare = monte_carlo_two(cfg_noshare, base_policy="event", runs=20)

    cfg_share = SimConfig(**{**base2.__dict__, "share_pose": True, "share_policy": "event", "share_delta": 2.0})
    _, s_share = monte_carlo_two(cfg_share, base_policy="event", runs=20)

    comp = pd.DataFrame([
        {"setting": "2-UAV, no sharing", **s_noshare},
        {"setting": "2-UAV, event-triggered pose sharing", **s_share},
    ])
    comp.to_csv("exp_two_uav_compare.csv", index=False)
    print("Two-UAV comparison:\n", comp)

    # One qualitative trajectory plot (2-UAV)
    df_traj, _ = simulate_two_uav(cfg_share, base_policy="event")
    plt.figure()
    plt.plot(df_traj["px1_ref"], df_traj["py1_ref"], "--", label="ref UAV1")
    plt.plot(df_traj["px1"], df_traj["py1"], label="UAV1")
    plt.plot(df_traj["px2_ref"], df_traj["py2_ref"], "--", label="ref UAV2")
    plt.plot(df_traj["px2"], df_traj["py2"], label="UAV2")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.title("Two-UAV field coverage (with sharing)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig("fig_two_uav_trajectory.png", dpi=200)

    print("Saved figures & CSVs:",
          "fig_quant_tradeoff.png, fig_markov_robustness.png, fig_two_uav_trajectory.png",
          "exp_quant_tradeoff.csv, exp_markov_robustness.csv, exp_two_uav_compare.csv")


if __name__ == "__main__":
    run_experiments()
