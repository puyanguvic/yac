# YAC: Event-Triggered SCC Simulations

This repository contains the simulation code and paper sources for a study on
event-triggered communication in sensing-communication-control (SCC) loops for
unmanned systems under intermittent wireless updates.

## Contents

- `content.tex`: Main paper LaTeX source.
- `src/yac_sim/`: Simulation package used to generate paper figures.
- `run_experiments.py`: Convenience entrypoint for running all experiments.
- `result/`: Generated figures and outputs (ignored by git).

## Requirements

- Python 3.10+
- Dependencies listed in `pyproject.toml` (numpy, scipy, matplotlib, pandas)

## Run experiments

Run all experiments and write figures to `result/`:

```bash
python run_experiments.py --outdir result
```

You can also run the module directly:

```bash
python -m yac_sim --outdir result
```

Quick run (reduced Monte Carlo runs and horizon):

```bash
python run_experiments.py --fast --outdir result
```

## Notes

- The simulations implement event-triggered, periodic, and random baselines.
- The paper text and figures are aligned to the "theory" and "robust" modes in
  `src/yac_sim/config.py`.
