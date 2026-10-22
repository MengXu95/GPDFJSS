import argparse
import contextlib
import io
import os
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from TransGP import experiment_config, main, main_GP_TransGP_test, main_model_train


SCENARIO = "Fmax-0.75-0.85-0.95-1.5"


@contextlib.contextmanager
def temporary_training_root(name, directory):
    previous = os.environ.get(name)
    os.environ[name] = str(directory)
    try:
        yield
    finally:
        if previous is None:
            del os.environ[name]
        else:
            os.environ[name] = previous


class ScenarioTests(unittest.TestCase):
    def test_canonical_task_configuration(self):
        scenario = experiment_config.parse_scenario(SCENARIO, "Fmax")
        experiment_config.require_transgp_tasks(scenario)
        self.assertEqual(scenario.utilizations, (0.75, 0.85, 0.95))

    def test_invalid_scenarios_are_rejected(self):
        for value in (
            "Fmax", "Unknown-0.75-1.5", "Fmax-nan-1.5", "Fmax-inf-1.5",
            "Fmax-1.0-1.5", "Fmax-0.0-1.5", "Fmax-0.75-2.0",
            "Fmax-0.750-1.5", "Fmax-abc-1.5",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                experiment_config.parse_scenario(value)
        with self.assertRaises(ValueError):
            experiment_config.parse_scenario(SCENARIO, "Tmean")

    def test_wrong_task_order_is_rejected(self):
        for value in ("Fmax-0.75-1.5", "Fmax-0.95-0.85-0.75-1.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                experiment_config.require_transgp_tasks(experiment_config.parse_scenario(value))

    def test_checkpoint_contract(self):
        paths = experiment_config.checkpoint_paths(SCENARIO)
        self.assertEqual(len(paths), 2)
        self.assertEqual(paths[0].parent.name, "sequencing_transformer_multitask")
        self.assertEqual(paths[1].parent.name, "routing_transformer_multitask")
        self.assertTrue(all(path.name == f"{SCENARIO}_diffusion_transformer_original.pt" for path in paths))


class EntryPointTests(unittest.TestCase):
    def assert_cli_error(self, entry, args):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            entry(args)
        self.assertEqual(error.exception.code, 2)

    def test_help_needs_no_neural_or_research_imports(self):
        with patch.dict(sys.modules, {"torch": None, "main_DRL": None}):
            for entry in (main.main, main_model_train.main, main_GP_TransGP_test.main):
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
                    entry(["--help"])
                self.assertEqual(error.exception.code, 0)

    def test_invalid_training_arguments(self):
        for args in (
            [SCENARIO, "Tmean", "0", "TransGP"],
            [SCENARIO, "Fmax", "-1", "TransGP"],
            [SCENARIO, "Fmax", "0", "TransGP", "10"],
            [SCENARIO, "Fmax", "0", "TransGP", "--wall-clock-seconds", "1"],
            [SCENARIO, "Fmax", "0", "MTGP"],
            ["Fmax-0.75-1.5", "Fmax", "0", "MTGP", "1"],
            ["Fmax-0.75-1.5", "Fmax", "0", "MTGP", "--wall-clock-seconds", "nan"],
            ["Fmax-0.75-1.5", "Fmax", "0", "MTGP", "--wall-clock-seconds", "-1"],
            [SCENARIO, "Fmax", "0", "CVAEGP"],
        ):
            with self.subTest(args=args):
                self.assert_cli_error(main.main, args)

    def test_missing_checkpoints_fail_before_evolution(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "TransGP.main.checkpoint_paths", return_value=(Path(directory) / "seq.pt", Path(directory) / "route.pt")
        ):
            self.assert_cli_error(main.main, [SCENARIO, "Fmax", "0", "TransGP"])

    def test_gp_dispatch_preserves_parameters(self):
        runner = Mock()
        module = SimpleNamespace(main=runner)
        with patch.dict(sys.modules, {"TransGP.archive_gp.GPFC": module}), patch("TransGP.main.os.chdir") as change_directory:
            main.main(["Fmax-0.75-1.5", "Fmax", "2", "MTGP", "6", "--wall-clock-seconds", "5"])
        change_directory.assert_called_once_with(experiment_config.PROJECT_ROOT)
        runner.assert_called_once_with(
            "Fmax-0.75-1.5", 2, "max-flowtime", 0.75, 6, wall_clock_limit_seconds=5.0,
        )

    def test_transgp_dispatch_preserves_task_order(self):
        runner = Mock()
        with patch.dict(sys.modules, {"TransGP.MultitaskTransformerGuidedMutationGP.GPFC": SimpleNamespace(main=runner)}), patch(
            "TransGP.main.checkpoint_paths", return_value=()
        ), patch("TransGP.main.os.chdir"):
            main.main([SCENARIO, "Fmax", "3", "TransGP"])
        runner.assert_called_once_with(SCENARIO, 3, "max-flowtime", [0.75, 0.85, 0.95])

    def test_transgp_test_dispatch_uses_single_objective_evaluator(self):
        evaluator = Mock()
        with patch.dict(sys.modules, {"TransGP.GP_TransGP_test": SimpleNamespace(test=evaluator)}), patch("TransGP.main.os.chdir"):
            main.main([SCENARIO, "Fmax", "0", "TransGP-test"])
        evaluator.assert_called_once_with("TransGP", SCENARIO, "Fmax", 0, [0.75, 0.85, 0.95])

    def test_test_entry_has_no_drl_dependency(self):
        evaluator = Mock()
        with patch.dict(sys.modules, {"TransGP.GP_TransGP_test": SimpleNamespace(test=evaluator), "main_DRL": None}), patch(
            "TransGP.main_GP_TransGP_test.os.chdir"
        ):
            main_GP_TransGP_test.main(["TransGP", SCENARIO, "Fmax", "0", "true"])
        evaluator.assert_called_once_with(
            "TransGP", SCENARIO, "Fmax", 0, [0.75, 0.85, 0.95], skip_existing=True,
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            main_GP_TransGP_test.parse_bool("maybe")

    def test_model_entry_rejects_unsupported_configuration(self):
        for args in (
            ["WFmean-0.75-0.85-0.95-1.5", "routing", "Transformer"],
            [SCENARIO, "routing", "Transformer", "0"],
            ["Fmax-0.75-1.5", "routing", "Transformer"],
            [SCENARIO, "routing", "LSTM"],
        ):
            with self.subTest(args=args):
                self.assert_cli_error(main_model_train.main, args)

    def test_model_entry_requires_all_archives(self):
        trainer = Mock()
        module = SimpleNamespace(main_train=trainer, objective_archive_seed_map=lambda: {"Fmax": [0, 1, 2]})
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            sys.modules, {"TransGP.Transformer.Transformer_tree_multitask": module}
        ), patch("TransGP.main_model_train.PROJECT_ROOT", Path(directory)), contextlib.chdir(directory):
            self.assert_cli_error(main_model_train.main, [SCENARIO, "sequencing", "Transformer", "1"])
        trainer.assert_not_called()

    def test_model_entry_saves_original_alias(self):
        trainer = Mock()
        module = SimpleNamespace(main_train=trainer, objective_archive_seed_map=lambda: {"Fmax": [0, 1, 2]})
        with patch.dict(sys.modules, {"TransGP.Transformer.Transformer_tree_multitask": module}), patch(
            "TransGP.main_model_train.Path.is_file", return_value=True
        ), patch("TransGP.main_model_train.os.chdir"):
            main_model_train.main([SCENARIO, "sequencing", "Transformer", "2"])
        trainer.assert_called_once_with(SCENARIO, "sequencing", save_original=True, epochs=2)


class SimulatorAndPersistenceTests(unittest.TestCase):
    def test_saved_rules_evaluate_on_all_three_tasks(self):
        import numpy as np
        import TransGP.GP_TransGP_test as evaluator

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            evaluator.METHOD_ROOTS, {"TransGP": Path(directory)}
        ):
            for utilization in (0.75, 0.85, 0.95):
                task = evaluator.scenario_for_task(SCENARIO, utilization)
                path = evaluator.candidate_files("TransGP", task, 0)[0]
                path.parent.mkdir(parents=True)
                with path.open("wb") as handle:
                    pickle.dump({0: [["PT"], ["WIQ"]]}, handle)
            original_shopfloor = evaluator.TestShopFloor

            def one_replication(objective, utilization, machines):
                return original_shopfloor(objective, utilization, machines, replications=1)

            with patch.object(evaluator, "TestShopFloor", side_effect=one_replication), contextlib.redirect_stdout(io.StringIO()):
                values = evaluator.test("TransGP", SCENARIO, "Fmax", 0, [0.75, 0.85, 0.95])
                cached = evaluator.test("TransGP", SCENARIO, "Fmax", 0, [0.75, 0.85, 0.95], skip_existing=True)
            self.assertEqual(values.shape, (3,))
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue((values > 0).all())
            np.testing.assert_array_equal(values, cached)
            fitness_path, runtime_path = evaluator.output_paths("TransGP", SCENARIO, 0)
            np.testing.assert_array_equal(np.load(fitness_path), values)
            self.assertEqual(np.load(runtime_path).shape, ())

    def test_missing_rules_fail_explicitly(self):
        import TransGP.GP_TransGP_test as evaluator

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            evaluator.METHOD_ROOTS, {"TransGP": Path(directory)}
        ), self.assertRaises(FileNotFoundError):
            evaluator.load_saved_individual("TransGP", "Fmax-0.75-1.5", 0, evaluator.build_pset())

    def test_candidate_priority_and_generation_selection(self):
        import TransGP.GP_TransGP_test as evaluator

        task = "Fmax-0.75-1.5"
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            evaluator.METHOD_ROOTS, {"TransGP": Path(directory)}
        ):
            final_path, generation_path, _ = evaluator.candidate_files("TransGP", task, 0)
            final_path.parent.mkdir(parents=True)
            for path in (final_path, generation_path):
                with path.open("wb") as handle:
                    pickle.dump({0: [["PT"], ["WIQ"]], 2: [["NPT"], ["WIQ"]]}, handle)
            pset = evaluator.build_pset()
            _, source, key = evaluator.load_saved_individual("TransGP", task, 0, pset)
            self.assertEqual((source, key), (final_path, 0))
            final_path.unlink()
            _, source, key = evaluator.load_saved_individual("TransGP", task, 0, pset)
            self.assertEqual((source, key), (generation_path, 2))
            with generation_path.open("wb") as handle:
                pickle.dump({}, handle)
            with self.assertRaises(ValueError):
                evaluator.load_saved_individual("TransGP", task, 0, pset)

    def test_gp_produces_nonempty_pretraining_archive(self):
        import numpy as np
        from deap import creator
        from TransGP.archive_gp import GPFC

        with tempfile.TemporaryDirectory() as directory, temporary_training_root(
            "MTGP_TRAIN_ROOT", directory
        ), patch.dict(creator.__dict__), patch.object(GPFC, "POP_SIZE", 32), patch.object(
            GPFC, "NGEN", 31
        ), contextlib.redirect_stdout(io.StringIO()):
            GPFC.main("Fmax-0.75-1.5", 123, "max-flowtime", 0.75)
            root = Path(directory) / "scenario_Fmax-0.75-1.5"
            values = np.load(root / "123_min_fitnessFmax-0.75-1.5.npy")
            self.assertEqual(values.shape, (32,))
            self.assertTrue(np.isfinite(values).all())
            with (root / "123_meng_archive_Fmax-0.75-1.5.pkl").open("rb") as handle:
                archive = pickle.load(handle)
            self.assertEqual(len(archive), 8)
            self.assertTrue(all(len(individual) == 2 for individual in archive.values()))


class TransformerSmokeTests(unittest.TestCase):
    def test_transgp_evolves_with_two_loaded_generators(self):
        import numpy as np
        import torch
        from deap import creator
        from TransGP.MultitaskTransformerGuidedMutationGP import GPFC, ea_simple_elitism
        from TransGP.Transformer.Transformer_tree_multitask import TransformerTree

        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, original_threads)
        original_loader = TransformerTree.load_model
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory), contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            for rule_type in ("sequencing", "routing"):
                tree = TransformerTree(777, SCENARIO, rule_type, max_len=128)
                tree.model_dir = str(root / rule_type)
                # Force short valid samples while exercising real inference and mutation.
                with torch.no_grad():
                    tree.model.fc_out.weight.zero_()
                    tree.model.fc_out.bias.fill_(-1000)
                    tree.model.fc_out.bias[tree.tokenizer.token_to_id["PT"]] = 1000
                tree.save_model(suffix="original")

            def load_temporary_model(tree, model_dir=None, epoch=None):
                return original_loader(tree, str(root / tree.rule_type), epoch)

            with patch.object(ea_simple_elitism.TransformerTree, "load_model", load_temporary_model), temporary_training_root(
                "TRANSGP_TRAIN_ROOT", root / "results" / "train"
            ), patch.dict(creator.__dict__), patch.object(GPFC, "POP_SIZE", 30), patch.object(GPFC, "NGEN", 1), patch.object(
                GPFC, "CXPB", 0.0
            ), patch.object(GPFC, "MUTPB", 1.0), patch.object(GPFC, "REPPB", 0.0):
                GPFC.main(SCENARIO, 123, "max-flowtime", [0.75, 0.85, 0.95])
            for utilization in (0.75, 0.85, 0.95):
                task = f"Fmax-{utilization}-1.5"
                task_root = root / "results" / "train" / f"scenario_{task}"
                values = np.load(task_root / f"123_min_fitness{task}.npy")
                self.assertEqual(values.shape, (2,))
                self.assertTrue(np.isfinite(values).all())
                self.assertTrue((task_root / f"123_meng_top_individuals_final_gen_{task}.pkl").is_file())

            import TransGP.GP_TransGP_test as evaluator

            with patch.dict(evaluator.METHOD_ROOTS, {"TransGP": root / "results"}), patch(
                "TransGP.main_GP_TransGP_test.os.chdir"
            ):
                main_GP_TransGP_test.main(["TransGP", SCENARIO, "Fmax", "123"])
                fitness_path, runtime_path = evaluator.output_paths("TransGP", SCENARIO, 123)
                values = np.load(fitness_path)
                self.assertEqual(values.shape, (3,))
                self.assertTrue(np.isfinite(values).all())
                self.assertTrue((values > 0).all())
                self.assertEqual(np.load(runtime_path).shape, ())

    def test_model_entry_trains_both_rule_types_from_archives(self):
        import torch
        from TransGP.Transformer.Transformer_tree_multitask import TransformerTree, objective_archive_seed_map

        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, original_threads)
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory), contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            for utilization in (0.75, 0.85, 0.95):
                task = f"Fmax-{utilization}-1.5"
                task_root = root / "archive_gp" / "train" / f"scenario_{task}"
                task_root.mkdir(parents=True)
                for seed in objective_archive_seed_map()["Fmax"]:
                    with (task_root / f"{seed}_meng_archive_{task}.pkl").open("wb") as handle:
                        pickle.dump({0: [["PT"], ["WIQ"]]}, handle)

            with patch("TransGP.main_model_train.PROJECT_ROOT", root), patch(
                "TransGP.Transformer.Transformer_tree_multitask.DrawTree.draw_tree"
            ):
                for rule_type in ("sequencing", "routing"):
                    main_model_train.main([SCENARIO, rule_type, "Transformer", "1"])
                    model = TransformerTree(777, SCENARIO, rule_type, max_len=128)
                    model.load_model()
                    checkpoint = (
                        root / "Transformer" / f"{rule_type}_transformer_multitask"
                        / f"{SCENARIO}_diffusion_transformer_original.pt"
                    )
                    self.assertTrue(checkpoint.is_file())
                    inputs = torch.tensor([[model.tokenizer.token_to_id["START"]]])
                    with torch.no_grad():
                        values = model.model(inputs, torch.tensor([[0.75, 6.0]]))
                    self.assertEqual(values.shape, (1, 1, model.tokenizer.vocab_size()))
                    self.assertTrue(torch.isfinite(values).all())

    def test_train_checkpoint_load_and_task_conditioning(self):
        import torch
        from TransGP.Transformer.Transformer_tree_multitask import TransformerTree

        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, original_threads)
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory), contextlib.redirect_stdout(io.StringIO()):
            tree = TransformerTree(777, SCENARIO, "sequencing", epochs=1, max_len=128)
            tree.model_dir = directory
            ids = tree.tokenizer.token_to_id
            inputs = torch.tensor([[ids["START"], ids["PT"]]])
            targets = torch.tensor([[ids["PT"], ids["END"]]])
            task = torch.tensor([[0.75, 6.0]])
            tree.train([(inputs, targets, task)], save_logs=False)
            self.assertTrue(torch.isfinite(torch.tensor(tree.logger.epoch_logs[0]["train_loss"])))
            tree.save_model(suffix="original")
            self.assertTrue((Path(directory) / f"{SCENARIO}_diffusion_transformer_original.pt").is_file())
            reloaded = TransformerTree(1, SCENARIO, "sequencing", max_len=128)
            reloaded.load_model(directory)
            with torch.no_grad():
                expected = reloaded.model(inputs, task)
                other_task = reloaded.model(inputs, torch.tensor([[0.95, 10.0]]))
                tree.model.eval()
                actual = tree.model(inputs, task)
            self.assertEqual(expected.shape, (1, 2, tree.tokenizer.vocab_size()))
            torch.testing.assert_close(actual, expected)
            self.assertFalse(torch.allclose(expected, other_task))


if __name__ == "__main__":
    unittest.main()
