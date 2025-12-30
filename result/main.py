from pathlib import Path

from yac_sim.experiments import run_experiments


def main():
    output_dir = Path("result")
    run_experiments(output_dir)


if __name__ == "__main__":
    main()
