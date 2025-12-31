from pathlib import Path

from .experiments import run_experiments


def main():
    output_dir = Path("result")
    run_experiments(output_dir, mode='paper')


if __name__ == "__main__":
    main()
