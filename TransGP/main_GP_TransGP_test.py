"""Evaluate saved GP or TransGP rules on independent simulator instances."""

import argparse
import os

from TransGP.experiment_config import OBJECTIVES, PROJECT_ROOT, parse_scenario, require_transgp_tasks


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
    try:
        scenario = parse_scenario(args.dataset_name, args.objective)
        if args.seed < 0:
            raise ValueError("Seed must be nonnegative")
        if args.method == "TransGP":
            require_transgp_tasks(scenario)
    except ValueError as error:
        parser.error(str(error))

    os.chdir(PROJECT_ROOT)
    from TransGP.GP_TransGP_test import test

    test(
        args.method, args.dataset_name, args.objective, args.seed,
        list(scenario.utilizations), skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
