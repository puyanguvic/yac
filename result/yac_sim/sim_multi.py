import math

import numpy as np
import pandas as pd

from .channels import GilbertElliottChannel
from .models import build_double_integrator, dlqr, generate_lawnmower_ref
from .quantization import uniform_quantize
from .utils import clip_u, packet_bits


def simulate_two_uav(cfg, base_policy: str):
    """
    Two UAVs each track a lawnmower path. Optionally share pose (px,py) between them
    under its own sharing trigger (event/periodic). Sharing consumes bits and can
    improve relative alignment. For simplicity, each UAV has its own channel instance
    but same parameters.
    """
    rng = np.random.default_rng(cfg.seed)
    A, B, _ = build_double_integrator(cfg.Ts)
    K, _ = dlqr(A, B, cfg.Q, cfg.R)

    ref1 = generate_lawnmower_ref(
        cfg.Ts,
        cfg.T_steps,
        cfg.field_L,
        cfg.field_W / 2,
        cfg.lane_spacing,
        cfg.v_ref,
        x0=0.0,
        y0=0.0,
    )
    ref2 = generate_lawnmower_ref(
        cfg.Ts,
        cfg.T_steps,
        cfg.field_L,
        cfg.field_W / 2,
        cfg.lane_spacing,
        cfg.v_ref,
        x0=0.0,
        y0=cfg.field_W / 2,
    )

    chan1 = GilbertElliottChannel(rng, cfg.p_loss_good, cfg.p_loss_bad, cfg.p_g2b, cfg.p_b2g, 0)
    chan2 = GilbertElliottChannel(rng, cfg.p_loss_good, cfg.p_loss_bad, cfg.p_g2b, cfg.p_b2g, 0)

    x1 = np.array([5.0, 0.0, 5.0, 0.0])
    x2 = np.array([5.0, 0.0, cfg.field_W / 2 + 5.0, 0.0])

    xhat1, xhat2 = x1.copy(), x2.copy()
    u1_prev, u2_prev = np.zeros(2), np.zeros(2)

    y1_prev = x1 + rng.normal(0.0, cfg.sigma_v, size=4)
    y2_prev = x2 + rng.normal(0.0, cfg.sigma_v, size=4)

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
        chan1.step()
        chan2.step()

        y1 = x1 + rng.normal(0.0, cfg.sigma_v, size=4)
        y2 = x2 + rng.normal(0.0, cfg.sigma_v, size=4)

        def decide_tx(policy, y, y_prev, k):
            if policy == "event":
                return np.linalg.norm(y - y_prev) > cfg.delta
            if policy == "periodic":
                return k % 1 == 0
            if policy == "random":
                return rng.random() < 1.0
            raise ValueError

        tx1 = decide_tx("event" if base_policy == "event" else "periodic", y1, y1_prev, k)
        tx2 = decide_tx("event" if base_policy == "event" else "periodic", y2, y2_prev, k)

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
            return (A @ xhat + B @ u_prev), y, False

        xhat1, y1_prev_new, rx1 = meas_update(tx1, chan1, y1, xhat1, u1_prev)
        xhat2, y2_prev_new, rx2 = meas_update(tx2, chan2, y2, xhat2, u2_prev)
        y1_prev = y1_prev_new if rx1 else y1_prev
        y2_prev = y2_prev_new if rx2 else y2_prev

        if cfg.share_pose:
            pose1 = x1[[0, 2]].copy()
            pose2 = x2[[0, 2]].copy()

            def share_decision(policy, pose, pose_prev, k):
                if policy == "event":
                    return np.linalg.norm(pose - pose_prev) > cfg.share_delta
                if policy == "periodic":
                    return k % cfg.share_period_M == 0
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
                shared1_from2 = pq1
                share_prev_1 = pose1.copy()
            if ok2:
                shared2_from1 = pq2
                share_prev_2 = pose2.copy()

        e1 = xhat1 - ref1[k]
        e2 = xhat2 - ref2[k]

        u1 = (-K @ e1.reshape(-1, 1)).flatten()
        u2 = (-K @ e2.reshape(-1, 1)).flatten()

        if cfg.share_pose:
            gain = 0.02
            dy1 = x1[2] - shared2_from1[1]
            dy2 = x2[2] - shared1_from2[1]
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

        rows.append(
            {
                "k": k,
                "px1": x1[0],
                "py1": x1[2],
                "px1_ref": ref1[k][0],
                "py1_ref": ref1[k][2],
                "err1": err1,
                "px2": x2[0],
                "py2": x2[2],
                "px2_ref": ref2[k][0],
                "py2_ref": ref2[k][2],
                "err2": err2,
                "tx_meas": int(tx1) + int(tx2),
                "tx_share": int(cfg.share_pose) * (0),
                "bits_used": int(bits_used),
                "chan1_state": int(chan1.state),
                "chan2_state": int(chan2.state),
            }
        )
        u1_prev, u2_prev = u1, u2

    df = pd.DataFrame(rows)
    rms1 = math.sqrt(float(np.mean(df["err1"].values**2)))
    rms2 = math.sqrt(float(np.mean(df["err2"].values**2)))
    return df, {
        "rms_err_uav1": rms1,
        "rms_err_uav2": rms2,
        "rms_err_avg": 0.5 * (rms1 + rms2),
        "bits_used": int(bits_used),
        "N_tx_meas": int(N_tx_meas),
        "N_tx_share": int(N_tx_share),
        "failed": int(failed),
    }
