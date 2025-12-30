import numpy as np


def uniform_quantize(x, bits, x_min=-10.0, x_max=10.0):
    """Uniform quantization for vector x with given bits per element."""
    if bits >= 32:
        return x.copy(), 0.0
    levels = 2**bits
    x_clipped = np.clip(x, x_min, x_max)
    step = (x_max - x_min) / (levels - 1)
    q = np.round((x_clipped - x_min) / step) * step + x_min
    qe = float(np.mean((q - x) ** 2))
    return q, qe
