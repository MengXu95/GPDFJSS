"""Evaluate saved GP or TransGP rules on independent simulator instances."""

import argparse

from main import main as run_main
from TransGP.experiment_config import OBJECTIVES


def parse_bool(value: str) -> bool:
    if value.lower() in ("1", "true", "yes", "y"):
        return True
    if value.lower() in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Use true or false for skip_existing")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("GP", "TransGP"))
    parser.add_argument("dataset_name")
    parser.add_argument("objective", choices=OBJECTIVES)
    parser.add_argument("seed", type=int)
    parser.add_argument("skip_existing", type=parse_bool, nargs="?", default=False)
    args = parser.parse_args(argv)
    algorithm = "TransGP-test" if args.method == "TransGP" else "TransGP-archive-test"
    options = [
        "--algorithm", algorithm, "--dataset-name", args.dataset_name,
        "--objective", args.objective, "--seed", str(args.seed),
    ]
    if args.skip_existing:
        options.append("--skip-existing")
    run_main(options)


if __name__ == "__main__":
    main()
