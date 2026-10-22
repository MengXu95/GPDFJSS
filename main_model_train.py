"""Train TransGP tree generators through the shared root-level entry point."""

import argparse
from contextlib import chdir
from pathlib import Path

from experiment_config import TRANSGP_ROOT, ModelTrainingConfig, load_configuration
from TransGP.experiment_config import parse_scenario


def run_training(config: ModelTrainingConfig, parser: argparse.ArgumentParser) -> None:
    from TransGP.Transformer.Transformer_tree_multitask import main_train, objective_archive_seed_map

    scenario = parse_scenario(config.dataset_name)
    with chdir(TRANSGP_ROOT):
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
        main_train(config.dataset_name, config.rule_type, save_original=True, epochs=config.epochs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON model-training configuration; named options override its values")
    parser.add_argument("--algorithm", choices=("TransGP",))
    parser.add_argument("--dataset-name")
    parser.add_argument("--rule-type", choices=("sequencing", "routing"))
    parser.add_argument("--model-type", choices=("Transformer",))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("legacy_args", nargs="*", help="Legacy: DATASET RULE_TYPE Transformer [EPOCHS]")
    args = parser.parse_args(argv)
    overrides = {
        name: value for name, value in vars(args).items()
        if name not in ("config", "legacy_args") and value is not None
    }
    data: dict[str, object]
    try:
        if args.legacy_args:
            if args.config or overrides or len(args.legacy_args) not in (3, 4):
                raise ValueError("Use either --config/named options or DATASET RULE_TYPE Transformer [EPOCHS]")
            dataset, rule_type, model_type, *epochs = args.legacy_args
            data = {
                "dataset_name": dataset, "rule_type": rule_type, "model_type": model_type,
                "epochs": int(epochs[0]) if epochs else 50,
            }
        else:
            data = load_configuration(args.config) if args.config else {}
            data.update(overrides)
        config = ModelTrainingConfig.from_mapping(data)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    run_training(config, parser)


if __name__ == "__main__":
    main()
