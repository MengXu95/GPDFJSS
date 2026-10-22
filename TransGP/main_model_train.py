"""Train one of the two task-conditioned Transformer tree generators."""

import argparse
import os
from pathlib import Path

from TransGP.experiment_config import PROJECT_ROOT, parse_scenario, require_transgp_tasks


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_name", help="For example Fmax-0.75-0.85-0.95-1.5")
    parser.add_argument("rule_type", choices=("sequencing", "routing"))
    parser.add_argument("model_type", choices=("Transformer",))
    parser.add_argument("epochs", nargs="?", type=int, default=50)
    args = parser.parse_args(argv)
    try:
        scenario = parse_scenario(args.dataset_name)
        require_transgp_tasks(scenario)
        if scenario.objective not in ("Fmax", "Fmean", "Tmean"):
            raise ValueError("Default archive seeds are defined only for Fmax, Fmean and Tmean")
        if args.epochs < 1:
            raise ValueError("Epoch count must be positive")
    except ValueError as error:
        parser.error(str(error))

    os.chdir(PROJECT_ROOT)
    from TransGP.Transformer.Transformer_tree_multitask import main_train, objective_archive_seed_map

    missing = []
    for utilization in scenario.utilizations:
        task_name = f"{scenario.objective}-{utilization}-1.5"
        for seed in objective_archive_seed_map()[scenario.objective]:
            path = (
                Path("archive_gp") / "train" / f"scenario_{task_name}"
                / f"{seed}_meng_archive_{task_name}.pkl"
            )
            if not path.is_file():
                missing.append(str(path))
    if missing:
        parser.error("Generate the single-task GP archives first. Missing archives:\n" + "\n".join(missing))
    main_train(args.dataset_name, args.rule_type, save_original=True, epochs=args.epochs)


if __name__ == "__main__":
    main()
