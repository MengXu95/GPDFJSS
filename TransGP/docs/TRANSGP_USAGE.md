# TransGP Usage Guide

This guide covers GP archive generation, task-conditioned Transformer training,
TransGP evolution and independent evaluation of the learned scheduling rules.
Use separate output directories for different experiments to avoid overwriting
models or results.

This is the single-objective TransGP integration inside GPDFJSS. Run commands
from the **GPDFJSS repository root** using `python -m TransGP.<entry>`.
The entry points put runtime artifacts under `TransGP`; paths shown in output
layouts below are relative to that directory unless stated otherwise.

The code comes from the source `main` commit
`bd01df0f81d635ed4c194a59801243a4ce8c86c0`. The shared simulator, ordinary GP
representation, primitive operators, archive reader, toolbox and logging
utilities are reused from GPDFJSS. Only the archive GP's differing evolution,
configuration and persistence modules are retained in `archive_gp`, leaving
the original baseline unchanged.

## 1. Overview

TransGP learns dispatching rules for multitask dynamic flexible job shop
scheduling (DFJSS). Each GP individual contains two program trees:

- **Sequencing:** selects the next waiting operation to process.
- **Routing:** selects the work center for an operation.

Ordinary GP runs generate historical archives. Two Transformers learn from the
sequencing and routing trees in those archives, respectively. During multitask
evolution, the models generate mutation subtrees conditioned on
`[utilization, machine count]`.

| Component | Implementation | Purpose |
| --- | --- | --- |
| Archive generation | [archive_gp](../archive_gp) | Single-task multitree GP; `MTGP` is the CLI name for this archive generator |
| Tree generators | [Transformer_tree_multitask.py](../Transformer/Transformer_tree_multitask.py) | Tree tokenization, task-conditioned Transformer training and checkpoints |
| TransGP setup | [GPFC.py](../MultitaskTransformerGuidedMutationGP/GPFC.py) | Task-tagged population, simulator fitness and parameters |
| Guided mutation | [multi_tree.py](../MultitaskTransformerGuidedMutationGP/multi_tree.py) | Sequencing/routing subtree generation using task conditions |
| Evolution | [ea_simple_elitism.py](../MultitaskTransformerGuidedMutationGP/ea_simple_elitism.py) | Model loading, selection, crossover, mutation and elitism |
| Simulator | [newSimulatorBasedOnJava](../../newSimulatorBasedOnJava) | Shared Python DFJSS simulator; no Java runtime is required |
| Evaluation | [GP_TransGP_test.py](../GP_TransGP_test.py) | Saved-rule loading, independent simulation and test results |

The paper reference is in the [project README](../../README.md#transgp-citation)
and [CITATION.cff](../CITATION.cff).

## 2. Installation

The following commands use PowerShell. Run them from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\TransGP\requirements.txt
$python = (Resolve-Path .\.venv\Scripts\python.exe).Path
```

The integration was validated with Python 3.11.9, NumPy 2.4.6, DEAP 1.4.4,
PyTorch 2.14.1, pandas 3.0.6 and graphviz 0.21.
[requirements.txt](../requirements.txt) specifies version ranges rather than an
exact environment lock.

Transformer training and inference currently run on **CPU**. Installing a
CUDA-enabled PyTorch build does not automatically enable GPU execution.

The example tree visualization at the end of model training also uses the
Graphviz `dot` executable. If it is unavailable, the drawing helper prints an
error; the model checkpoints saved before visualization remain available.

Training logs use Excel when `openpyxl` is installed, otherwise they are saved
as CSV with a message indicating the output paths. Excel logging is optional:

```powershell
.\.venv\Scripts\python.exe -m pip install openpyxl
```

## 3. Scenarios and default settings

Scenario names follow `<objective>-<utilization...>-<due-date factor>`.
Examples:

- Single task: `Fmax-0.75-1.5`
- TransGP: `Fmax-0.75-0.85-0.95-1.5`

The default TransGP task conditions, in order, are `[0.75, 6]`, `[0.85, 8]` and
`[0.95, 10]`. Archive generation, model training and evaluation must use matching
machine counts. The scenario string alone does not define arbitrary task
embeddings.

| Parameter | Default |
| --- | --- |
| Recorded jobs / warm-up jobs | 124 / 0 |
| Due-date factor | 1.5 |
| Replications per training evaluation | 1 |
| Initial training simulation seed | 968356, rotated each generation |
| Ordinary GP population | 200 |
| TransGP total population | 600 |
| Generations | 50, with 51 records including generation 0 |
| Crossover / mutation / reproduction | 0.80 / 0.15 / 0.05 |
| Tournament size / elitism / maximum tree height | 4 / 4 / 6 |
| Transformer | `d_model=256`, 4 heads, 4 decoder layers, dropout 0.1 |
| Model training | Seed 777, batch size 128, learning rate 0.001, 50 epochs, `max_len=128` |
| Independent evaluation | Initial simulation seed 77777, 30 replications |

The scheduling objective identifiers are:

| Identifier | Objective |
| --- | --- |
| `Fmax` | Maximum flowtime |
| `Fmean` | Mean flowtime |
| `WFmean` | Mean weighted flowtime |
| `Tmax` | Maximum tardiness |
| `Tmean` | Mean tardiness |
| `WTmean` | Mean weighted tardiness |

**Default pretraining archive seed mappings are defined only for `Fmax`,
`Fmean` and `Tmean`.** Training a generator for another objective requires a
corresponding archive seed mapping and matching archives. The standard TransGP
CLI also requires the three ordered tasks above.

## 4. Generate GP archives

The following example generates all nine archives required for `Fmax`:

```powershell
foreach ($util in @("0.75", "0.85", "0.95")) {
    foreach ($seed in 0..2) {
        & $python -m TransGP.main "Fmax-$util-1.5" Fmax $seed MTGP
        if ($LASTEXITCODE -ne 0) { throw "GP archive generation failed" }
    }
}
```

Machine counts default to 6, 8 and 10 for utilization levels 0.75, 0.85 and 0.95,
respectively. If a machine count is supplied explicitly to the GP CLI, it must
match the task condition used to train the Transformer.

The model trainer reads these archive seeds for **each** utilization level:

| Objective | Archive seeds |
| --- | --- |
| `Fmax` | 0, 1, 2 |
| `Fmean` | 3, 4, 5 |
| `Tmean` | 6, 7, 8 |

For example, to generate the `Fmean` dataset, replace `Fmax` with `Fmean` and
`0..2` with `3..5` in the loop above.

Archive files use the following layout:

```text
archive_gp\train\scenario_Fmax-0.75-1.5\
    0_meng_archive_Fmax-0.75-1.5.pkl
```

The trainer needs three utilization levels times three archive seeds, not one
archive under a combined multitask scenario.

**Archives are collected only from generation 31 onward.** A shortened or
time-limited run can produce an empty archive and cannot replace a complete
pretraining dataset. Each GP run defaults to 50 generations.

Do not use the root-level basic GP entry to generate these archives: it retains
its original 50-individual, five-generation defaults and ten-machine setting.
The TransGP archive CLI also accepts an optional machine count and a
generation-boundary time budget, for example:

```powershell
& $python -m TransGP.main Fmax-0.75-1.5 Fmax 0 MTGP 6 --wall-clock-seconds 60
```

The budget is checked between generations, not during a simulation. This
shortened run is not a substitute for a full pretraining archive.

## 5. Train both Transformers

After the required archives exist, train the sequencing and routing models:

```powershell
& $python -m TransGP.main_model_train Fmax-0.75-0.85-0.95-1.5 sequencing Transformer 50
if ($LASTEXITCODE -ne 0) { throw "Sequencing model training failed" }
& $python -m TransGP.main_model_train Fmax-0.75-0.85-0.95-1.5 routing Transformer 50
if ($LASTEXITCODE -ne 0) { throw "Routing model training failed" }
```

TransGP loads these two checkpoints:

```text
Transformer\sequencing_transformer_multitask\Fmax-0.75-0.85-0.95-1.5_diffusion_transformer_original.pt
Transformer\routing_transformer_multitask\Fmax-0.75-0.85-0.95-1.5_diffusion_transformer_original.pt
```

`diffusion_transformer` is a historical filename, not an indication that this
implementation uses a diffusion model. The training CLI saves the `original`
checkpoint aliases expected by evolution; numbered checkpoints alone do not
satisfy the default loading paths.

Each objective requires its own sequencing and routing models. GP archives and
pretrained weights are not bundled with the repository.

## 6. Run TransGP

```powershell
& $python -m TransGP.main Fmax-0.75-0.85-0.95-1.5 Fmax 0 TransGP
```

To run multiple independent GP seeds:

```powershell
foreach ($seed in 0..29) {
    & $python -m TransGP.main Fmax-0.75-0.85-0.95-1.5 Fmax $seed TransGP
    if ($LASTEXITCODE -ne 0) { throw "TransGP failed at seed $seed" }
}
```

This is an example batch of 30 runs. Model-training seeds, archive-generation
seeds, GP-run seeds and evaluation simulation seeds have separate roles.
Full training runs can take substantially longer than the automated tests.

## 7. Evaluate saved rules

```powershell
& $python -m TransGP.main_GP_TransGP_test TransGP Fmax-0.75-0.85-0.95-1.5 Fmax 0
```

For each task, the evaluator first looks for the final-generation top-individual
file and selects its smallest key. If that file is absent, it tries the
per-generation individual file and then the archive, selecting the largest key.
It does not select a rule by comparing test fitness across candidates.

Evaluation uses the corresponding 6/8/10-machine tasks and, by default, 30
independent simulation replications starting from seed 77777. The resulting
NumPy fitness array contains three values in scenario utilization order:

```text
MultitaskTransformerGuidedMutationGP\train\scenario_Fmax-0.75-0.85-0.95-1.5\
    0_test_fitnessFmax-0.75-0.85-0.95-1.5.npy
    0_test_running_timeFmax-0.75-0.85-0.95-1.5.npy
```

These are final-rule evaluation results, not per-generation test curves.

The alternate command below runs the same TransGP evaluator:

```powershell
& $python -m TransGP.main Fmax-0.75-0.85-0.95-1.5 Fmax 0 TransGP-test
```

To evaluate one saved archive-GP rule instead, use method `GP` and a single-task
scenario. This reads the isolated archive-GP outputs, not the root baseline:

```powershell
& $python -m TransGP.main_GP_TransGP_test GP Fmax-0.75-1.5 Fmax 0
```

## 8. Outputs and configuration

TransGP writes training results separately for each task:

```text
MultitaskTransformerGuidedMutationGP\train\scenario_Fmax-0.75-1.5\
    0_meng_individual_Fmax-0.75-1.5.pkl
    0_meng_top_individuals_final_gen_Fmax-0.75-1.5.pkl
    0_min_fitnessFmax-0.75-1.5.npy
    0_running_timeFmax-0.75-1.5.npy
    0_Fmax-0.75-1.5_each_gen.txt
```

The fitness array records the best training fitness per generation. Runtime is
the duration of the entire multitask run, saved for each task; do not sum the
three copies as if they were separate runs.

| Environment variable | Effect |
| --- | --- |
| `TRANSGP_TRAIN_ROOT` | TransGP training output directory |
| `TRANSGP_METHOD_ROOT` | Evaluation method root; saved rules must be under `<root>\train\scenario_<task>` |
| `MTGP_CXPB`, `MTGP_MUTPB`, `MTGP_REPPB` | TransGP operator probabilities; defaults are 0.80, 0.15 and 0.05 |
| `MTGP_POP_SIZE`, `MTGP_TOURSIZE` | Ordinary GP population and tournament size; these do not change the TransGP population |

Relative custom paths are resolved from `TransGP`, because the CLI changes its
working directory there. Prefer absolute custom output paths.
`MTGP_TRAIN_ROOT` optionally relocates archive-GP outputs only; the model trainer
still expects archives under `TransGP\archive_gp\train`, so leave it unset for
the standard workflow.

When using a custom TransGP output directory, set `TRANSGP_TRAIN_ROOT` to
`<root>\train` and `TRANSGP_METHOD_ROOT` to `<root>` so training and evaluation
use matching paths. Transformer checkpoints remain under `Transformer` and are
not relocated by these variables.

Running the same method, scenario and seed again can overwrite existing
outputs. Model weights, archives, NumPy results, logs and virtual environments
are excluded from Git tracking. Load pickle and model files only from trusted
sources.

## 9. Automated tests

```powershell
& $python -m unittest discover -s .\TransGP\tests -t . -v
```

The tests cover source and import boundaries, CLI validation, GP archive
generation, model training and checkpoint loading, guided evolution and final
rule evaluation. They create temporary fixtures and do not require downloaded
archives or pretrained models.

The GP archive test runs 32 individuals through generation 31 and checks for
eight archived two-tree individuals. Model-entry tests train the sequencing and
routing generators using nine temporary archive fixtures. The guided-evolution
test uses synthetic checkpoints, 30 individuals and one mutation-only
generation, then evaluates the saved rules with the default 30 replications.
These reduced-scale tests exercise functionality; they do not reproduce the
paper's full experimental results.
