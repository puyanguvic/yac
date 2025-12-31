import math

import numpy as np
import pandas as pd

from .channels import GilbertElliottChannel
from .models import build_double_integrator, dlqr, generate_lawnmower_ref
from .quantization import uniform_quantize
from .utils import clip_u, packet_bits


def simulate_single_uav(cfg, policy: str, periodic_M=1, random_q=1.0):
    """Simulate a single UAV tracking a lawnmower reference under communication constraints.

    This function supports two analysis modes:

    - cfg.mode == "theory":
        Matches the paper baseline abstraction:
          * perfect measurement y_k = x_k
          * if a transmission is attempted, it is delivered (no loss)
          * upon receiving an update, the controller-side estimate is reset to truth
            (x_hat_k = x_k), yielding an explicit grow-and-reset prediction error.

    - cfg.mode == "robust":
        Uses measurement noise, quantization, and Gilbert–Elliott packet loss.

    Policies:
      - 'event': send if ||tilde_pred|| > delta, where tilde_pred = x - x_hat_pred
      - 'periodic': send every M steps
      - 'random': send with prob q
    """
    # Normalize mode switches if user didn't set the flags explicitly
    if hasattr(cfg, "normalize_modes"):
        cfg.normalize_modes()

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
    x_hat = np.zeros(4)
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

        # ---- measurement (paper baseline uses perfect state) ----
        if cfg.ideal_measurement:
            y = (C @ x)
        else:
            y = (C @ x) + rng.normal(0.0, cfg.sigma_v, size=4)

        # ---- controller-side prediction ----
        x_hat_pred = A @ x_hat + B @ u_prev
        tilde_pred = x - x_hat_pred  # prediction error before (potential) update

        # ---- bits per value (optional adaptive coding) ----
        bpv = cfg.bits_per_value
        if getattr(cfg, "adaptive_bits", False):
            bpv = 6 if chan.state == 1 else cfg.bits_per_value

        # ---- decide whether to transmit ----
        if policy == "event":
            do_tx = np.linalg.norm(tilde_pred) > cfg.delta
        elif policy == "periodic":
            do_tx = (k % periodic_M) == 0
        elif policy == "random":
            do_tx = rng.random() < random_q
        else:
            raise ValueError("Unknown policy")

        # ---- transmission + reception ----
        received = False
        loss_p = cfg.p_loss_good if chan.state == 0 else cfg.p_loss_bad

        if do_tx:
            pkt_bits = packet_bits(4, bpv, cfg.bits_per_packet_overhead)
            if bits_used + pkt_bits <= cfg.bit_budget_total:
                bits_used += pkt_bits
                N_tx_attempt += 1

                if cfg.ideal_comm:
                    received = True
                else:
                    received, _, _ = chan.transmit()
            else:
                do_tx = False

        # ---- estimator update (grow-and-reset lives here) ----
        if received:
            N_tx_delivered += 1

            if cfg.ideal_quant:
                x_hat = x.copy()  # reset to truth (paper baseline)
            else:
                # In the robust mode, we treat the payload as the measurement (possibly quantized).
                yq, qe = uniform_quantize(y, bits=bpv, x_min=-50.0, x_max=50.0)
                qerr_acc += qe
                x_hat = yq.copy()
        else:
            x_hat = x_hat_pred

        # After update (or not), define the realized prediction error
        tilde = x - x_hat

        # ---- control ----
        e_hat = x_hat - x_ref[k]
        u = (-K @ e_hat.reshape(-1, 1)).flatten()
        u = clip_u(u, cfg.u_max)

        e_true = x - x_ref[k]
        J_cost += float(e_true.T @ cfg.Q @ e_true + u.T @ cfg.R @ u)

        # ---- plant step ----
        x = A @ x + B @ u
        if getattr(cfg, "sigma_w", 0.0) > 0.0:
            x = x + rng.normal(0.0, cfg.sigma_w, size=4)
        u_prev = u

        # ---- failure metric ----
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
                "px": float(x[0]),
                "py": float(x[2]),
                "px_ref": float(x_ref[k][0]),
                "py_ref": float(x_ref[k][2]),
                "pos_err": pos_err,
                "x_norm": float(np.linalg.norm(x)),
                "tilde_pred_norm": float(np.linalg.norm(tilde_pred)),
                "tilde_norm": float(np.linalg.norm(tilde)),
                "ux": float(u[0]),
                "uy": float(u[1]),
                "tx": int(do_tx),
                "rx": int(received),
                "chan_state": int(chan.state),
                "loss_p": float(loss_p),
                "bits_used": int(bits_used),
                "bpv": int(bpv),
            }
        )

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
