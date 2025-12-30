import numpy as np
from scipy.linalg import solve_discrete_are


def build_double_integrator(Ts: float):
    A = np.array(
        [
            [1.0, Ts, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, Ts],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    B = np.array(
        [
            [0.5 * Ts * Ts, 0.0],
            [Ts, 0.0],
            [0.0, 0.5 * Ts * Ts],
            [0.0, Ts],
        ]
    )
    C = np.eye(4)
    return A, B, C


def dlqr(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
    return K, P


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
