"""Compatibility adapter for the former TransGP experiment CLI."""

import argparse

from main import main as run_main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_name")
    parser.add_argument("objective")
    parser.add_argument("seed", type=int)
    parser.add_argument("algo", choices=("MTGP", "TransGP", "TransGP-test"))
    parser.add_argument("machine_num", nargs="?", type=int)
    parser.add_argument("--wall-clock-seconds", type=float)
    args = parser.parse_args(argv)
    algorithm = "TransGP-archive" if args.algo == "MTGP" else args.algo
    options = [
        "--algorithm", algorithm, "--dataset-name", args.dataset_name,
        "--objective", args.objective, "--seed", str(args.seed),
    ]
    if args.machine_num is not None:
        options.extend(("--machine-num", str(args.machine_num)))
    if args.wall_clock_seconds is not None:
        options.extend(("--wall-clock-seconds", str(args.wall_clock_seconds)))
    run_main(options)


if __name__ == "__main__":
    main()
