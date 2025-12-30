import math

import numpy as np
import pandas as pd

from .channels import GilbertElliottChannel
from .models import build_double_integrator, dlqr, generate_lawnmower_ref
from .quantization import uniform_quantize
from .utils import clip_u, packet_bits


def simulate_single_uav(cfg, policy: str, periodic_M=1, random_q=1.0):
    """
    policy:
      - 'event': send if ||y-x_hat_pred|| > delta
      - 'periodic': send every M steps
      - 'random': send with prob q
    channel: Gilbert-Elliott
    budget: bit_budget_total
    payload: send 4 values (state measurement)
    """
    rng = np.random.default_rng(cfg.seed)
    A, B, C = build_double_integrator(cfg.Ts)
    K, _ = dlqr(A, B, cfg.Q, cfg.R)
    x_ref = generate_lawnmower_ref(
        cfg.Ts,
        cfg.T_steps,
        cfg.field_L,
        cfg.field_W,
        cfg.lane_spacing,
        cfg.v_ref,
    )

    chan = GilbertElliottChannel(
        rng,
        p_loss_good=cfg.p_loss_good,
        p_loss_bad=cfg.p_loss_bad,
        p_g2b=cfg.p_g2b,
        p_b2g=cfg.p_b2g,
        init_state=0,
    )

    x = np.zeros(4)
    x[0], x[2] = 5.0, 5.0
    x_hat = x.copy()
    u_prev = np.zeros(2)

    bits_used = 0
    N_tx_attempt = 0
    N_tx_delivered = 0
    qerr_acc = 0.0
    J_cost = 0.0

    consecutive_bad = 0
    failed = False

    rows = []

    for k in range(cfg.T_steps):
        chan.step()
        y = (C @ x) + rng.normal(0.0, cfg.sigma_v, size=4)

        x_hat_pred = A @ x_hat + B @ u_prev

        bpv = cfg.bits_per_value
        if cfg.adaptive_bits:
            bpv = 6 if chan.state == 1 else cfg.bits_per_value

        do_tx = False
        if policy == "event":
            do_tx = np.linalg.norm(y - x_hat_pred) > cfg.delta
        elif policy == "periodic":
            do_tx = k % periodic_M == 0
        elif policy == "random":
            do_tx = rng.random() < random_q
        else:
            raise ValueError("Unknown policy")

        received = False
        loss_p = cfg.p_loss_good if chan.state == 0 else cfg.p_loss_bad

        if do_tx:
            pkt_bits = packet_bits(4, bpv, cfg.bits_per_packet_overhead)
            if bits_used + pkt_bits <= cfg.bit_budget_total:
                bits_used += pkt_bits
                N_tx_attempt += 1
                received, _, _ = chan.transmit()
            else:
                do_tx = False

        if received:
            N_tx_delivered += 1
            yq, qe = uniform_quantize(y, bits=bpv, x_min=-50.0, x_max=50.0)
            qerr_acc += qe
            x_hat = yq.copy()
        else:
            x_hat = x_hat_pred

        tilde = x - x_hat
        x_norm = float(np.linalg.norm(x))
        tilde_norm = float(np.linalg.norm(tilde))

        e_hat = x_hat - x_ref[k]
        u = (-K @ e_hat.reshape(-1, 1)).flatten()
        u = clip_u(u, cfg.u_max)

        e_true = x - x_ref[k]
        J_cost += float(e_true.T @ cfg.Q @ e_true + u.T @ cfg.R @ u)

        x = A @ x + B @ u

        pos_err = float(np.linalg.norm(x[[0, 2]] - x_ref[k][[0, 2]]))
        if pos_err > cfg.fail_err:
            consecutive_bad += 1
        else:
            consecutive_bad = 0
        if consecutive_bad >= cfg.fail_window:
            failed = True

        rows.append(
            {
                "k": k,
                "px": x[0],
                "py": x[2],
                "px_ref": x_ref[k][0],
                "py_ref": x_ref[k][2],
                "pos_err": pos_err,
                "x_norm": x_norm,
                "tilde_norm": tilde_norm,
                "ux": u[0],
                "uy": u[1],
                "tx": int(do_tx),
                "rx": int(received),
                "chan_state": int(chan.state),
                "loss_p": float(loss_p),
                "bits_used": int(bits_used),
                "bpv": int(bpv),
            }
        )
        u_prev = u

    df = pd.DataFrame(rows)
    rms = math.sqrt(float(np.mean(df["pos_err"].values**2)))
    energy = float(np.sum(df["ux"].values**2 + df["uy"].values**2))
    tx_rate = float(N_tx_delivered) / cfg.T_steps
    tx_attempt_rate = float(N_tx_attempt) / cfg.T_steps
    avg_qerr = float(qerr_acc / max(1, df["rx"].sum()))
    return df, {
        "rms_err": rms,
        "energy": energy,
        "J_cost": float(J_cost),
        "N_tx": int(N_tx_delivered),
        "N_tx_attempt": int(N_tx_attempt),
        "tx_rate": tx_rate,
        "tx_attempt_rate": tx_attempt_rate,
        "bits_used": int(bits_used),
        "avg_qerr": avg_qerr,
        "failed": int(failed),
    }
