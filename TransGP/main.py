"""Run ordinary GP archive generation or task-conditioned TransGP."""

import argparse
import math
import os

from TransGP.experiment_config import (
    OBJECTIVES,
    PROJECT_ROOT,
    checkpoint_paths,
    parse_scenario,
    require_transgp_tasks,
    resolve_objective,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_name", help="For example Fmax-0.75-0.85-0.95-1.5")
    parser.add_argument("objective", choices=OBJECTIVES)
    parser.add_argument("seed", type=int)
    parser.add_argument("algo", choices=("MTGP", "TransGP", "TransGP-test"))
    parser.add_argument("machine_num", nargs="?", type=int, help="MTGP only; defaults to 6/8/10 by task")
    parser.add_argument("--wall-clock-seconds", type=float, help="MTGP only; generation-boundary time budget")
    args = parser.parse_args(argv)
    try:
        scenario = parse_scenario(args.dataset_name, args.objective)
        if args.seed < 0:
            raise ValueError("Seed must be nonnegative")
        if args.algo == "MTGP":
            if len(scenario.utilizations) != 1:
                raise ValueError("MTGP archive generation requires a single-task scenario")
            if args.machine_num is not None and args.machine_num < 2:
                raise ValueError("The heterogeneous simulator requires at least two machines")
            if args.wall_clock_seconds is not None and (
                not math.isfinite(args.wall_clock_seconds) or args.wall_clock_seconds <= 0
            ):
                raise ValueError("--wall-clock-seconds must be finite and positive")
        else:
            require_transgp_tasks(scenario)
            if args.machine_num is not None or args.wall_clock_seconds is not None:
                raise ValueError("Machine overrides and wall-clock budgets are supported only for MTGP")
            if args.algo == "TransGP":
                missing = [str(path) for path in checkpoint_paths(args.dataset_name) if not path.is_file()]
                if missing:
                    raise ValueError("Train both Transformer models first. Missing checkpoints:\n" + "\n".join(missing))
    except ValueError as error:
        parser.error(str(error))

    os.chdir(PROJECT_ROOT)
    objective = resolve_objective(args.objective)
    if args.algo == "MTGP":
        from TransGP.archive_gp.GPFC import main as run_gp

        run_gp(
            args.dataset_name, args.seed, objective, scenario.utilizations[0],
            args.machine_num, wall_clock_limit_seconds=args.wall_clock_seconds,
        )
    elif args.algo == "TransGP":
        from TransGP.MultitaskTransformerGuidedMutationGP.GPFC import main as run_transgp

        run_transgp(args.dataset_name, args.seed, objective, list(scenario.utilizations))
    else:
        from TransGP.GP_TransGP_test import test

        test("TransGP", args.dataset_name, args.objective, args.seed, list(scenario.utilizations))


if __name__ == "__main__":
    main()
