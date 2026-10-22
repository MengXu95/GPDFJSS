"""Run basic GP or TransGP experiments using JSON configurations."""

import argparse
from contextlib import chdir

from experiment_config import (
    ALGORITHMS, PROJECT_ROOT, TRANSGP_ROOT, ExperimentConfig, load_configuration,
)
from TransGP.experiment_config import checkpoint_paths, resolve_objective


def run_experiment(config: ExperimentConfig) -> None:
    root = PROJECT_ROOT if config.algorithm in ("MTGP", "MTGP-test") else TRANSGP_ROOT
    with chdir(root):
        objective = resolve_objective(config.objective)
        if config.algorithm == "MTGP":
            from MTGPforNewSimulatorBasedOnJava.GPFC import main as run_gp

            run_gp(config.dataset_name, config.seed, objective, config.utilizations[0])
        elif config.algorithm == "MTGP-test":
            from Test.testAndResultsSave import main as test_gp

            test_gp(
                "MTGPforNewSimulatorBasedOnJava", config.dataset_name, config.seed,
                objective, config.utilizations[0], 10,
            )
        elif config.algorithm == "TransGP-archive":
            from TransGP.archive_gp.GPFC import main as generate_archives

            generate_archives(
                config.dataset_name, config.seed, objective, config.utilizations[0],
                config.machine_num, wall_clock_limit_seconds=config.wall_clock_seconds,
            )
        elif config.algorithm == "TransGP":
            from TransGP.MultitaskTransformerGuidedMutationGP.GPFC import main as run_transgp

            run_transgp(config.dataset_name, config.seed, objective, list(config.utilizations))
        else:
            from TransGP.GP_TransGP_test import test

            method = "GP" if config.algorithm == "TransGP-archive-test" else "TransGP"
            test(
                method, config.dataset_name, config.objective, config.seed,
                list(config.utilizations), skip_existing=config.skip_existing,
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON configuration file; named options override its values")
    parser.add_argument("--algorithm", choices=ALGORITHMS)
    parser.add_argument("--dataset-name")
    parser.add_argument("--objective")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--utilization", type=float)
    parser.add_argument("--machine-num", type=int)
    parser.add_argument("--wall-clock-seconds", type=float)
    parser.add_argument("--skip-existing", action="store_true", default=None)
    parser.add_argument("legacy_args", nargs="*", help="Legacy: DATASET OBJECTIVE UTILIZATION SEED MTGP|MTGP-test")
    args = parser.parse_args(argv)
    overrides = {
        name: value for name, value in vars(args).items()
        if name not in ("config", "legacy_args") and value is not None
    }
    data: dict[str, object]
    try:
        if args.legacy_args:
            if args.config or overrides or len(args.legacy_args) != 5:
                raise ValueError("Use either --config/named options or five legacy positional arguments")
            dataset, objective, utilization, seed, algorithm = args.legacy_args
            if algorithm not in ("MTGP", "MTGP-test"):
                raise ValueError("Legacy positional syntax supports only MTGP and MTGP-test; use --algorithm")
            data = {
                "dataset_name": dataset, "objective": objective, "utilization": float(utilization),
                "seed": int(seed), "algorithm": algorithm,
            }
        else:
            data = load_configuration(args.config) if args.config else {}
            data.update(overrides)
        config = ExperimentConfig.from_mapping(data)
        if config.algorithm == "TransGP":
            missing = [str(path) for path in checkpoint_paths(config.dataset_name) if not path.is_file()]
            if missing:
                raise ValueError("Train both Transformer models first. Missing checkpoints:\n" + "\n".join(missing))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    run_experiment(config)


if __name__ == "__main__":
    main()
