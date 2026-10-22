import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import experiment_config as configuration
import main
import main_model_train
from TransGP import main_model_train as legacy_model_entry


ROOT = Path(__file__).resolve().parents[2]
SCENARIO = "Fmax-0.75-0.85-0.95-1.5"
GP_CONFIG = {
    "algorithm": "MTGP", "dataset_name": "Fmax-0.85-1.5", "objective": "Fmax", "seed": 0,
}
MODEL_CONFIG = {
    "algorithm": "TransGP", "dataset_name": SCENARIO, "rule_type": "sequencing",
    "model_type": "Transformer", "epochs": 50,
}


class ConfigurationTests(unittest.TestCase):
    def test_all_example_files_validate(self):
        paths = list((ROOT / "configs").glob("*.json"))
        self.assertEqual(len(paths), 8)
        for path in paths:
            with self.subTest(path=path.name):
                data = configuration.load_configuration(path)
                config_type = (
                    configuration.ModelTrainingConfig if path.name.startswith("transformer")
                    else configuration.ExperimentConfig
                )
                config_type.from_mapping(data)

    def test_json_loader_rejects_malformed_ambiguous_or_nonobject_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            for text in (
                "[]", "null", "1", '{"algorithm":', '{"seed": 0, "seed": 1}',
                '{"seed": NaN}', '{"seed": Infinity}',
            ):
                with self.subTest(text=text):
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        configuration.load_configuration(path)
            path.write_text(json.dumps(GP_CONFIG), encoding="utf-8-sig")
            self.assertEqual(configuration.load_configuration(path), GP_CONFIG)

    def test_experiment_validation_rejects_wrong_types_and_irrelevant_fields(self):
        invalid_overrides = (
            {"algorithm": "Unknown"}, {"algorithm": None}, {"dataset_name": []},
            {"objective": "Tmean"}, {"seed": -1}, {"seed": True}, {"seed": "1"},
            {"seed": 1.0}, {"sead": 1}, {"utilization": 0.75}, {"utilization": True},
            {"utilization": None}, {"machine_num": 10}, {"wall_clock_seconds": 1},
            {"skip_existing": False}, {"epochs": 1},
            {"algorithm": "TransGP"}, {"dataset_name": SCENARIO},
            {"algorithm": "TransGP-archive", "machine_num": 1},
            {"algorithm": "TransGP-archive", "machine_num": 6.0},
            {"algorithm": "TransGP-archive", "wall_clock_seconds": 0},
            {"algorithm": "TransGP-archive", "wall_clock_seconds": float("inf")},
            {"algorithm": "TransGP-test", "dataset_name": SCENARIO, "skip_existing": "false"},
        )
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                configuration.ExperimentConfig.from_mapping({**GP_CONFIG, **overrides})
        for field in ("algorithm", "dataset_name", "objective"):
            data = dict(GP_CONFIG)
            del data[field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                configuration.ExperimentConfig.from_mapping(data)

    def test_model_validation_rejects_wrong_types_and_unavailable_methods(self):
        for overrides in (
            {"algorithm": "MTGP"}, {"model_type": "CVAE"}, {"model_type": "LSTM"},
            {"rule_type": "unknown"}, {"epochs": 0}, {"epochs": True}, {"epochs": "1"},
            {"seed": 1}, {"dataset_name": "Fmax-0.75-1.5"},
            {"dataset_name": "WFmean-0.75-0.85-0.95-1.5"},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                configuration.ModelTrainingConfig.from_mapping({**MODEL_CONFIG, **overrides})

    def test_test_configuration_cannot_be_used_for_model_training(self):
        with self.assertRaises(ValueError):
            configuration.ModelTrainingConfig.from_mapping(
                configuration.load_configuration(ROOT / "configs" / "transgp_train.json")
            )
        with self.assertRaises(ValueError):
            configuration.ExperimentConfig.from_mapping(MODEL_CONFIG)


class UnifiedEntryTests(unittest.TestCase):
    def assert_cli_error(self, entry, args):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            entry(args)
        self.assertEqual(error.exception.code, 2)

    def test_each_experiment_configuration_dispatches_with_correct_arguments_and_directory(self):
        cases = (
            ("gp_train", "MTGPforNewSimulatorBasedOnJava.GPFC", "main",
             ("Fmax-0.85-1.5", 0, "max-flowtime", 0.85), {}, ROOT),
            ("gp_test", "Test.testAndResultsSave", "main",
             ("MTGPforNewSimulatorBasedOnJava", "Fmax-0.85-1.5", 0, "max-flowtime", 0.85, 10), {}, ROOT),
            ("transgp_archive", "TransGP.archive_gp.GPFC", "main",
             ("Fmax-0.75-1.5", 0, "max-flowtime", 0.75, None),
             {"wall_clock_limit_seconds": None}, ROOT / "TransGP"),
            ("transgp_train", "TransGP.MultitaskTransformerGuidedMutationGP.GPFC", "main",
             (SCENARIO, 0, "max-flowtime", [0.75, 0.85, 0.95]), {}, ROOT / "TransGP"),
            ("transgp_test", "TransGP.GP_TransGP_test", "test",
             ("TransGP", SCENARIO, "Fmax", 0, [0.75, 0.85, 0.95]),
             {"skip_existing": False}, ROOT / "TransGP"),
            ("transgp_archive_test", "TransGP.GP_TransGP_test", "test",
             ("GP", "Fmax-0.75-1.5", "Fmax", 0, [0.75]),
             {"skip_existing": False}, ROOT / "TransGP"),
        )
        original_directory = Path.cwd()
        for name, module, function, expected_args, expected_kwargs, expected_root in cases:
            runner = Mock(side_effect=lambda *args, **kwargs: self.assertEqual(Path.cwd(), expected_root))
            with self.subTest(name=name), patch.dict(
                sys.modules, {module: SimpleNamespace(**{function: runner})}
            ), patch("main.checkpoint_paths", return_value=()):
                main.main(["--config", str(ROOT / "configs" / f"{name}.json")])
                runner.assert_called_once_with(*expected_args, **expected_kwargs)
                self.assertEqual(Path.cwd(), original_directory)

    def test_cli_overrides_configuration_without_changing_file(self):
        runner = Mock()
        path = ROOT / "configs" / "transgp_archive.json"
        before = path.read_bytes()
        with patch.dict(sys.modules, {"TransGP.archive_gp.GPFC": SimpleNamespace(main=runner)}):
            main.main([
                "--config", str(path), "--dataset-name", "Fmean-0.95-1.5",
                "--objective", "Fmean", "--seed", "5", "--machine-num", "10",
                "--wall-clock-seconds", "12.5",
            ])
        runner.assert_called_once_with(
            "Fmean-0.95-1.5", 5, "mean-flowtime", 0.95, 10, wall_clock_limit_seconds=12.5,
        )
        self.assertEqual(path.read_bytes(), before)

    def test_legacy_root_gp_and_named_options_are_equivalent(self):
        runner = Mock()
        with patch.dict(sys.modules, {"MTGPforNewSimulatorBasedOnJava.GPFC": SimpleNamespace(main=runner)}):
            main.main(["Fmax-0.85-1.5", "max-flowtime", "0.85", "3", "MTGP"])
            main.main([
                "--algorithm", "MTGP", "--dataset-name", "Fmax-0.85-1.5",
                "--objective", "Fmax", "--seed", "3",
            ])
        self.assertEqual(runner.call_count, 2)
        self.assertEqual(runner.call_args_list[0], runner.call_args_list[1])

    def test_model_configuration_overrides_and_compatibility_adapter(self):
        self.assertIs(legacy_model_entry.main, main_model_train.main)
        trainer = Mock()
        module = SimpleNamespace(main_train=trainer, objective_archive_seed_map=lambda: {"Fmax": [0, 1, 2]})
        original_directory = Path.cwd()
        with patch.dict(sys.modules, {"TransGP.Transformer.Transformer_tree_multitask": module}), patch(
            "main_model_train.Path.is_file", return_value=True
        ):
            main_model_train.main([
                "--config", str(ROOT / "configs" / "transformer_sequencing.json"),
                "--rule-type", "routing", "--epochs", "1",
            ])
        trainer.assert_called_once_with(SCENARIO, "routing", save_original=True, epochs=1)
        self.assertEqual(Path.cwd(), original_directory)

    def test_invalid_files_or_mixed_syntax_fail_as_cli_errors(self):
        for entry in (main.main, main_model_train.main):
            with self.subTest(entry=entry.__module__):
                self.assert_cli_error(entry, ["--config", str(ROOT / "no-such-config.json")])
                self.assert_cli_error(entry, [])
                self.assert_cli_error(entry, ["--config", str(ROOT / "configs" / "gp_train.json"), "extra"])
        self.assert_cli_error(main.main, ["--config", str(ROOT / "configs" / "transformer_sequencing.json")])
        self.assert_cli_error(main_model_train.main, ["--config", str(ROOT / "configs" / "transgp_train.json")])

    def test_algorithm_failure_restores_callers_working_directory(self):
        runner = Mock(side_effect=RuntimeError("simulator failure"))
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory), patch.dict(
            sys.modules, {"TransGP.archive_gp.GPFC": SimpleNamespace(main=runner)}
        ):
            with self.assertRaisesRegex(RuntimeError, "simulator failure"):
                main.main(["--config", str(ROOT / "configs" / "transgp_archive.json")])
            self.assertEqual(Path.cwd(), Path(directory))

    def test_missing_model_archives_restore_callers_working_directory(self):
        trainer = Mock()
        module = SimpleNamespace(main_train=trainer, objective_archive_seed_map=lambda: {"Fmax": [0, 1, 2]})
        with tempfile.TemporaryDirectory() as directory, patch(
            "main_model_train.TRANSGP_ROOT", Path(directory)
        ), patch.dict(sys.modules, {"TransGP.Transformer.Transformer_tree_multitask": module}):
            original_directory = Path.cwd()
            self.assert_cli_error(
                main_model_train.main, ["--config", str(ROOT / "configs" / "transformer_routing.json")]
            )
            self.assertEqual(Path.cwd(), original_directory)
        trainer.assert_not_called()

    def test_root_help_needs_no_model_dependencies(self):
        with patch.dict(sys.modules, {"torch": None, "conditionalVAE": None, "LSTMDecoder": None}):
            for entry in (main.main, main_model_train.main):
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
                    entry(["--help"])
                self.assertEqual(error.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
