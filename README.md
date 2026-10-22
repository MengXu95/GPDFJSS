# GPDFJSS: GP and TransGP for DFJSS

This project provides two approaches to **dynamic flexible job shop scheduling
(DFJSS)** in Python:

- **Basic GP:** DEAP-based multi-tree genetic programming (the `MTGP` command).
  Each individual evolves a sequencing tree and a routing tree for one scenario.
- **TransGP:** task-conditioned Transformer-guided GP. Two pretrained tree
  generators guide mutation across multiple scheduling tasks. This integration
  is **single-objective**, not multi-objective: every task optimizes the same
  chosen objective.

Both approaches use the shared [Python DFJSS simulator](newSimulatorBasedOnJava),
developed based on the Java simulator in [GPJSS](https://github.com/meiyi1986/GPJSS).
No Java runtime is required.

## Layout and integration boundaries

| Path | Purpose |
| --- | --- |
| [main.py](main.py) | Unified JSON-configured GP/TransGP training and evaluation CLI |
| [main_model_train.py](main_model_train.py) | Unified Transformer model-pretraining CLI |
| [configs](configs) | Ready-to-run configurations for each workflow stage |
| [experiment_config.py](experiment_config.py) | Shared configuration loading and validation |
| [MTGPforNewSimulatorBasedOnJava](MTGPforNewSimulatorBasedOnJava) | Original GP implementation and shared primitives, operators and archive loader |
| [newSimulatorBasedOnJava](newSimulatorBasedOnJava) | Shared DFJSS simulator |
| [Test](Test) | Original basic GP evaluation |
| [TransGP](TransGP) | Single-objective TransGP archive generator, Transformers, evolution and tests |
| [TransGP/archive_gp](TransGP/archive_gp) | GP variant used to generate TransGP pretraining archives |
| [TransGP/Transformer](TransGP/Transformer) | Task-conditioned tree generators; checkpoints are generated here |
| [TransGP/MultitaskTransformerGuidedMutationGP](TransGP/MultitaskTransformerGuidedMutationGP) | TransGP population, guided mutation and saved rules |
| [Utils](Utils) | Shared rule-size and fitness-diversity utilities |
| [paintingAnalysis](paintingAnalysis), [wilcoxonTest](wilcoxonTest) | Existing plotting and statistical analysis tools |

TransGP was imported from the `main` branch of
[MengXu95/TransGP](https://github.com/MengXu95/TransGP), commit
`bd01df0f81d635ed4c194a59801243a4ce8c86c0`. Only its single-objective workflow
is included, without other methods, experiment results, archives, pretrained
weights or virtual environments.

The simulator and compatible GP helpers are reused rather than copied.
TransGP-specific imports are namespaced under `TransGP`, and default outputs
stay inside that directory. The original basic GP algorithm parameters and
output locations are unchanged. Root-level entry points now handle both
approaches; the old TransGP entry modules are compatibility adapters, not
separate implementations.

## Installation

Use Python 3.11. The examples below use PowerShell and must be run from the
**GPDFJSS repository root**, not from inside the TransGP directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\TransGP\requirements.txt
$python = (Resolve-Path .\.venv\Scripts\python.exe).Path
```

The requirements cover the basic GP and TransGP workflows. Transformer training
and inference currently run on **CPU**. Graphviz's optional `dot` executable
enables tree images; optional `openpyxl` enables Excel logs. See the
[usage guide](TransGP/docs/TRANSGP_USAGE.md#2-installation) for details.

## Unified configuration

Choose a JSON configuration to trigger the required algorithm or stage:

| Configuration | Entry point | `algorithm` / stage |
| --- | --- | --- |
| [gp_train.json](configs/gp_train.json) | `main.py` | `MTGP`: original basic GP training |
| [gp_test.json](configs/gp_test.json) | `main.py` | `MTGP-test`: original basic GP evaluation |
| [transgp_archive.json](configs/transgp_archive.json) | `main.py` | `TransGP-archive`: generate a single-task pretraining archive |
| [transgp_archive_test.json](configs/transgp_archive_test.json) | `main.py` | `TransGP-archive-test`: evaluate an archive-GP rule |
| [transgp_train.json](configs/transgp_train.json) | `main.py` | `TransGP`: guided multitask evolution |
| [transgp_test.json](configs/transgp_test.json) | `main.py` | `TransGP-test`: independent evaluation |
| [transformer_sequencing.json](configs/transformer_sequencing.json) | `main_model_train.py` | `TransGP` / sequencing model pretraining |
| [transformer_routing.json](configs/transformer_routing.json) | `main_model_train.py` | `TransGP` / routing model pretraining |

For example, the TransGP evolution configuration is:

```json
{
  "algorithm": "TransGP",
  "dataset_name": "Fmax-0.75-0.85-0.95-1.5",
  "objective": "Fmax",
  "seed": 0
}
```

Run it after generating the required archives and both checkpoints:

```powershell
& $python main.py --config .\configs\transgp_train.json
```

Named CLI options override the JSON values without modifying the file:

```powershell
& $python main.py --config .\configs\transgp_train.json --seed 7
& $python main_model_train.py --config .\configs\transformer_routing.json --epochs 20
& $python main.py --help
& $python main_model_train.py --help
```

### Configuration fields

For experiments, `algorithm`, `dataset_name` and `objective` are required.
`seed` defaults to 0 and must be a nonnegative integer. `objective` accepts the
short identifiers (`Fmax`, `Fmean`, etc.) or simulator names such as
`max-flowtime`; it must agree with the scenario name.

| Optional experiment field | Supported use |
| --- | --- |
| `utilization` | Single-task runs only; if provided, must match `dataset_name`. Otherwise inferred from it |
| `machine_num` | `TransGP-archive` only; integer >= 2, otherwise inferred as 6/8/10 from utilization |
| `wall_clock_seconds` | `TransGP-archive` only; finite positive generation-boundary budget |
| `skip_existing` | Boolean for `TransGP-test` or `TransGP-archive-test`; defaults to false |

Model configurations use `dataset_name`, `rule_type` (`sequencing` or `routing`),
`algorithm` (`TransGP`), `model_type` (`Transformer`) and `epochs` (positive
integer, default 50). Model pretraining does not apply to basic GP. CVAE/LSTM
are not included; the former root trainer's imports of those missing modules
have been removed.

Use the matching entry point for each configuration. Unknown fields, invalid
types, duplicate keys and incompatible task settings fail explicitly rather
than being ignored. Config files are read relative to the caller's working
directory **before** execution changes directory. Basic GP still writes under
the repository root; all TransGP stages still write under `TransGP`.
The caller's working directory is restored after execution, including failures.
These configurations select workflows and entry-point arguments; algorithm
population/generation defaults remain in their implementation modules.

## Basic GP for DFJSS

```powershell
& $python main.py --config .\configs\gp_train.json
if ($LASTEXITCODE -ne 0) { throw "Basic GP training failed" }
& $python main.py --config .\configs\gp_test.json
```

The basic GP uses ten machines and currently defaults to population 50 and
5 generations. In `Fmax-0.85-1.5`, `0.85` is utilization and the trailing
`1.5` is the due-date factor. Training
outputs are saved under
`MTGPforNewSimulatorBasedOnJava\train\scenario_<dataset>`.

## TransGP for DFJSS

The complete workflow is:

```text
Single-task GP archives
    -> sequencing + routing Transformer pretraining
    -> multitask TransGP evolution
    -> independent evaluation of saved rules
```

Use the same two root-level entry points, selecting the TransGP configurations
below. Unlike `MTGP`, `TransGP-archive` uses the archive-generation settings
needed for pretraining.

### 1. Generate the GP archives

TransGP uses the ordered task pairs `(utilization, machines)` of `(0.75, 6)`,
`(0.85, 8)` and `(0.95, 10)`. For `Fmax`, generate all nine archives:

```powershell
foreach ($util in @("0.75", "0.85", "0.95")) {
    foreach ($seed in 0..2) {
        & $python main.py --config .\configs\transgp_archive.json --dataset-name "Fmax-$util-1.5" --seed $seed
        if ($LASTEXITCODE -ne 0) { throw "GP archive generation failed" }
    }
}
```

This command uses the isolated [archive GP](TransGP/archive_gp), with population
200 and 50 generations, not the original five-generation basic GP.
Archives are collected only after generation 30, so a short basic GP run
cannot replace them. Archives are saved under
`TransGP\archive_gp\train\scenario_<single-task>`.

### 2. Train both tree generators

```powershell
& $python main_model_train.py --config .\configs\transformer_sequencing.json
if ($LASTEXITCODE -ne 0) { throw "Sequencing model training failed" }
& $python main_model_train.py --config .\configs\transformer_routing.json
if ($LASTEXITCODE -ne 0) { throw "Routing model training failed" }
```

Each model reads all nine GP archives. The resulting checkpoints are:

```text
TransGP\Transformer\sequencing_transformer_multitask\Fmax-0.75-0.85-0.95-1.5_diffusion_transformer_original.pt
TransGP\Transformer\routing_transformer_multitask\Fmax-0.75-0.85-0.95-1.5_diffusion_transformer_original.pt
```

`diffusion_transformer` is a historical filename; these are Transformer models.
Archives and pretrained weights are **not bundled** and must be generated first.

### 3. Evolve and evaluate TransGP rules

```powershell
& $python main.py --config .\configs\transgp_train.json
if ($LASTEXITCODE -ne 0) { throw "TransGP evolution failed" }
& $python main.py --config .\configs\transgp_test.json
```

TransGP defaults to a total population of 600 and 50 generations. Training
results are written to
`TransGP\MultitaskTransformerGuidedMutationGP\train\scenario_<single-task>`.
Evaluation uses 30 independent replications and saves a three-value fitness
array, ordered by utilization, under the combined scenario directory.

Default archive seed mappings are available for **Fmax** (0, 1, 2), **Fmean**
(3, 4, 5) and **Tmean** (6, 7, 8). Each objective needs its own archives and both
checkpoints. Full experiments are not quick smoke tests.

See the [TransGP usage guide](TransGP/docs/TRANSGP_USAGE.md) for parameters,
saved-file formats, optional archive-GP budgets and custom output paths.
Running the same scenario and seed again can overwrite its outputs. Only load
pickle archives and model checkpoints from trusted sources.

## Legacy command compatibility

Existing positional commands remain supported. Their implementations now
delegate to the unified root entry points:

```powershell
# Original basic GP: DATASET OBJECTIVE UTILIZATION SEED ALGO.
& $python main.py Fmax-0.85-1.5 max-flowtime 0.85 0 MTGP

# Former TransGP archive command: MTGP here maps to TransGP-archive.
& $python -m TransGP.main Fmax-0.75-1.5 Fmax 0 MTGP

# Both old model-training commands reach the same root implementation.
& $python main_model_train.py Fmax-0.75-0.85-0.95-1.5 routing Transformer 50
& $python -m TransGP.main_model_train Fmax-0.75-0.85-0.95-1.5 routing Transformer 50

# Former evaluation command also delegates to main.py.
& $python -m TransGP.main_GP_TransGP_test TransGP Fmax-0.75-0.85-0.95-1.5 Fmax 0
```

Do not mix positional syntax with `--config` or the new named options.
Use the JSON configurations for new experiments.

## Tests

```powershell
& $python -m unittest discover -s .\TransGP\tests -t . -v
```

Tests use temporary fixtures and reduced-scale experiments, covering archive
generation, both model trainers, guided mutation/evolution, independent
evaluation, shared imports and preservation of the original GP configuration.
They also cover JSON validation, all example configurations, CLI overrides,
legacy commands and working-directory restoration.
No pre-existing archives or checkpoints are required.

## TransGP citation

Meng Xu, Jiao Liu, Hua Yu, and Yew Soon Ong.
**TransGP: Task-Conditioned Transformer-Guided Genetic Programming for Multitask
Dynamic Flexible Job Shop Scheduling.**
*IEEE Transactions on Evolutionary Computation*, Early Access, 2026.
DOI: [10.1109/TEVC.2026.3734959](https://doi.org/10.1109/TEVC.2026.3734959).
Machine-readable citation metadata is in [TransGP/CITATION.cff](TransGP/CITATION.cff).
