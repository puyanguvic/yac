# YAC Simulation (Refactored)

This folder contains the refactored simulation code and experiment runner. The code is split into small modules under `result/yac_sim`, with a simple entry point in `result/main.py`.

## Structure

- `result/main.py`: runs all experiments and writes outputs into `result/`.
- `result/yac_sim/config.py`: simulation configuration dataclass.
- `result/yac_sim/models.py`: system model and reference generation.
- `result/yac_sim/channels.py`: Gilbert-Elliott channel model.
- `result/yac_sim/quantization.py`: uniform quantizer.
- `result/yac_sim/utils.py`: shared helpers.
- `result/yac_sim/sim_single.py`: single-UAV simulation.
- `result/yac_sim/sim_multi.py`: two-UAV simulation.
- `result/yac_sim/experiments.py`: Monte Carlo and plotting.

## Run

```bash
python result/main.py
```

Outputs will be saved to `result/`:

- `exp_quant_tradeoff.csv`
- `exp_markov_robustness.csv`
- `exp_two_uav_compare.csv`
- `fig_quant_tradeoff.png`
- `fig_markov_robustness.png`
- `fig_two_uav_trajectory.png`
