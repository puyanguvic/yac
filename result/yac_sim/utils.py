import numpy as np


def clip_u(u, u_max):
    n = np.linalg.norm(u)
    if n <= u_max:
        return u
    return u * (u_max / (n + 1e-12))


def packet_bits(num_values, bits_per_value, overhead_bits):
    return overhead_bits + num_values * bits_per_value
