import ast
import contextlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from TransGP import experiment_config


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent


class IntegrationScopeTests(unittest.TestCase):
    def test_shared_components_are_not_copied(self):
        for name in ("newSimulatorBasedOnJava", "Utils", "MTGPforNewSimulatorBasedOnJava"):
            with self.subTest(name=name):
                self.assertFalse((ROOT / name).exists())
        self.assertEqual(
            {path.name for path in (ROOT / "archive_gp").glob("*.py")},
            {"__init__.py", "GPFC.py", "ea_simple_elitism.py", "saveFile.py"},
        )
        self.assertFalse((ROOT / "MultitaskTransformerGuidedMutationGP" / "ParallelToolbox.py").exists())

    def test_python_sources_have_complete_import_boundaries(self):
        shared = {
            "newSimulatorBasedOnJava", "MTGPforNewSimulatorBasedOnJava", "Utils",
            "main", "main_model_train", "experiment_config",
        }
        external = sys.stdlib_module_names | {"numpy", "deap", "torch", "pandas", "graphviz"}
        for path in ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                else:
                    continue
                for module in modules:
                    name = module.split(".")[0]
                    with self.subTest(path=str(path.relative_to(ROOT)), module=module):
                        if name == "TransGP" or name in shared:
                            target = PROJECT_ROOT.joinpath(*module.split("."))
                            self.assertTrue(target.with_suffix(".py").is_file() or target.is_dir())
                        else:
                            self.assertIn(name, external)

    def test_checkpoints_and_working_directory_are_isolated(self):
        self.assertEqual(experiment_config.PROJECT_ROOT, ROOT)
        paths = experiment_config.checkpoint_paths("Fmax-0.75-0.85-0.95-1.5")
        self.assertTrue(all(path.is_relative_to(ROOT / "Transformer") for path in paths))

    def test_module_entry_points_work_from_repository_root(self):
        for module in (
            "main", "main_model_train", "TransGP.main",
            "TransGP.main_model_train", "TransGP.main_GP_TransGP_test",
        ):
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-m", module, "--help"],
                    cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_documentation_links_resolve(self):
        for path in (PROJECT_ROOT / "README.md", ROOT / "docs" / "TRANSGP_USAGE.md"):
            links = re.findall(r"\[[^\]]*\]\(([^)\s]+)\)", path.read_text(encoding="utf-8"))
            for link in links:
                if "://" in link or link.startswith("#"):
                    continue
                target = link.split("#", 1)[0]
                with self.subTest(document=path.name, link=link):
                    self.assertTrue((path.parent / target).exists())


class SharedGPRegressionTests(unittest.TestCase):
    def test_original_gp_configuration_and_helpers_are_preserved(self):
        from MTGPforNewSimulatorBasedOnJava import GPFC as baseline, multi_tree
        from MTGPforNewSimulatorBasedOnJava.ParallelToolbox import ParallelToolbox
        from TransGP.archive_gp import GPFC as archives
        from TransGP.MultitaskTransformerGuidedMutationGP import GPFC as guided

        self.assertEqual((baseline.POP_SIZE, baseline.NGEN, baseline.ELITISM), (50, 5, 10))
        self.assertEqual((baseline.CXPB, baseline.MUTPB), (0.8, 0.2))
        self.assertEqual((archives.POP_SIZE, archives.NGEN), (200, 50))
        self.assertEqual((guided.POP_SIZE, guided.NGEN), (600, 50))
        self.assertIs(archives.REP, multi_tree)
        self.assertIs(archives.ParallelToolbox, ParallelToolbox)
        self.assertIs(guided.ParallelToolbox, ParallelToolbox)
        self.assertIs(guided.REP.init_primitives, multi_tree.init_primitives)
        self.assertIs(guided.REP.lim_xmate, multi_tree.lim_xmate)
        self.assertIs(guided.REP.wrap, multi_tree.wrap)
        self.assertIsNot(archives.ea_simple_elitism, baseline.ea_simple_elitism)
        self.assertIsNot(archives.saveFile, baseline.saveFile)
        self.assertEqual(baseline.shopfloor().numMachines, 10)
        self.assertEqual(archives.shopfloor(numMachines=6).maxNumOperations, 6)

    def test_original_gp_still_trains_and_saves_rules(self):
        import numpy as np
        from deap import creator
        from MTGPforNewSimulatorBasedOnJava import GPFC
        import main

        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory), patch.dict(
            creator.__dict__
        ), patch.object(GPFC, "POP_SIZE", 12), patch.object(GPFC, "NGEN", 1), patch.object(
            GPFC, "ELITISM", 4
        ), contextlib.redirect_stdout(io.StringIO()):
            config_path = Path(directory) / "gp.json"
            config_path.write_text(json.dumps({
                "algorithm": "MTGP", "dataset_name": "Fmax-0.85-1.5",
                "objective": "Fmax", "seed": 123,
            }), encoding="utf-8")
            with patch("main.PROJECT_ROOT", Path(directory)):
                main.main(["--config", str(config_path)])
            root = Path(directory) / "MTGPforNewSimulatorBasedOnJava" / "train" / "scenario_Fmax-0.85-1.5"
            values = np.load(root / "123_min_fitnessFmax-0.85-1.5.npy")
            self.assertEqual(values.shape, (2,))
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue((root / "123_meng_individual_Fmax-0.85-1.5.pkl").is_file())


if __name__ == "__main__":
    unittest.main()
