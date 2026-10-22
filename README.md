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
| [main.py](main.py) | Original basic GP training and testing CLI |
| [MTGPforNewSimulatorBasedOnJava](MTGPforNewSimulatorBasedOnJava) | Original GP implementation and shared primitives, operators and archive loader |
| [newSimulatorBasedOnJava](newSimulatorBasedOnJava) | Shared DFJSS simulator |
| [Test](Test) | Original basic GP evaluation |
| [TransGP](TransGP) | Single-objective TransGP CLI, archive generator, Transformers, evolution and tests |
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
stay inside that directory. The original basic GP entry point, algorithm
parameters and output locations are unchanged.

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

## Basic GP for DFJSS

The original CLI takes five positional arguments:

```text
python main.py DATASET OBJECTIVE UTILIZATION SEED ALGO
```

```powershell
& $python main.py Fmax-0.85-1.5 max-flowtime 0.85 0 MTGP
& $python main.py Fmax-0.85-1.5 max-flowtime 0.85 0 MTGP-test
```

Here `0.85` is utilization; the trailing `1.5` in the dataset name is the
due-date factor, **not** the utilization argument. The basic GP uses ten
machines and currently defaults to population 50 and 5 generations. Training
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

Use module entry points to avoid collisions with the original root-level
scripts:

```powershell
& $python -m TransGP.main --help
& $python -m TransGP.main_model_train --help
& $python -m TransGP.main_GP_TransGP_test --help
```

### 1. Generate the GP archives

TransGP uses the ordered task pairs `(utilization, machines)` of `(0.75, 6)`,
`(0.85, 8)` and `(0.95, 10)`. For `Fmax`, generate all nine archives:

```powershell
foreach ($util in @("0.75", "0.85", "0.95")) {
    foreach ($seed in 0..2) {
        & $python -m TransGP.main "Fmax-$util-1.5" Fmax $seed MTGP
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
& $python -m TransGP.main_model_train Fmax-0.75-0.85-0.95-1.5 sequencing Transformer 50
if ($LASTEXITCODE -ne 0) { throw "Sequencing model training failed" }
& $python -m TransGP.main_model_train Fmax-0.75-0.85-0.95-1.5 routing Transformer 50
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
& $python -m TransGP.main Fmax-0.75-0.85-0.95-1.5 Fmax 0 TransGP
if ($LASTEXITCODE -ne 0) { throw "TransGP evolution failed" }
& $python -m TransGP.main_GP_TransGP_test TransGP Fmax-0.75-0.85-0.95-1.5 Fmax 0
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

## Tests

```powershell
& $python -m unittest discover -s .\TransGP\tests -t . -v
```

Tests use temporary fixtures and reduced-scale experiments, covering archive
generation, both model trainers, guided mutation/evolution, independent
evaluation, shared imports and preservation of the original GP configuration.
No pre-existing archives or checkpoints are required.

## TransGP citation

Meng Xu, Jiao Liu, Hua Yu, and Yew Soon Ong.
**TransGP: Task-Conditioned Transformer-Guided Genetic Programming for Multitask
Dynamic Flexible Job Shop Scheduling.**
*IEEE Transactions on Evolutionary Computation*, Early Access, 2026.
DOI: [10.1109/TEVC.2026.3734959](https://doi.org/10.1109/TEVC.2026.3734959).
Machine-readable citation metadata is in [TransGP/CITATION.cff](TransGP/CITATION.cff).
