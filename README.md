# GPDFJSS: GP and TransGP for DFJSS

Python implementations of genetic programming for **dynamic flexible job shop
scheduling (DFJSS)**:

- **Basic GP (`MTGP`):** evolves a sequencing tree and a routing tree for one
  scheduling scenario.
- **TransGP:** uses task-conditioned Transformers to guide GP mutation across
  multiple scheduling tasks. It is **single-objective**: all tasks optimize
  the same chosen objective.

Both methods share a [Python DFJSS simulator](newSimulatorBasedOnJava), developed
based on [GPJSS](https://github.com/meiyi1986/GPJSS). No Java runtime is required.

Start with [installation](#installation), then follow either the
[basic GP](#basic-gp-for-dfjss) or [TransGP](#transgp-for-dfjss) workflow.

## Installation

Use Python 3.11. Run all examples below in PowerShell from the
**GPDFJSS repository root**:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\TransGP\requirements.txt
$python = (Resolve-Path .\.venv\Scripts\python.exe).Path
```

These dependencies cover both methods. Transformer training and inference run
on **CPU**. Graphviz's `dot` executable and `openpyxl` are optional; see the
[installation details](TransGP/docs/TRANSGP_USAGE.md#2-installation).

## Basic GP for DFJSS

Train a rule, then evaluate it:

```powershell
& $python main.py --config .\configs\gp_train.json
if ($LASTEXITCODE -ne 0) { throw "Basic GP training failed" }
& $python main.py --config .\configs\gp_test.json
```

The example optimizes maximum flowtime at utilization 0.85 with ten machines.
Basic GP defaults to population 50 and 5 generations. Outputs are saved under
`MTGPforNewSimulatorBasedOnJava\train\scenario_<dataset>`.

## TransGP for DFJSS

Run these stages **in order**:

```text
GP archives -> sequencing + routing Transformers -> TransGP evolution -> evaluation
```

Archives and pretrained weights are **not bundled**. The example below uses
`Fmax` (maximum flowtime) and the ordered `(utilization, machines)` tasks
`(0.75, 6)`, `(0.85, 8)` and `(0.95, 10)`. Full experiments can take
substantially longer than the automated tests.

### 1. Generate all nine GP archives

```powershell
foreach ($util in @("0.75", "0.85", "0.95")) {
    foreach ($seed in 0..2) {
        & $python main.py --config .\configs\transgp_archive.json --dataset-name "Fmax-$util-1.5" --seed $seed
        if ($LASTEXITCODE -ne 0) { throw "GP archive generation failed" }
    }
}
```

`TransGP-archive` uses population 200 and 50 generations. Archives are collected
only after generation 30; the short basic `MTGP` run cannot replace this stage.
They are saved under `TransGP\archive_gp\train\scenario_<single-task>`.

### 2. Train both tree generators

```powershell
& $python main_model_train.py --config .\configs\transformer_sequencing.json
if ($LASTEXITCODE -ne 0) { throw "Sequencing model training failed" }
& $python main_model_train.py --config .\configs\transformer_routing.json
if ($LASTEXITCODE -ne 0) { throw "Routing model training failed" }
```

Each model reads all nine archives and trains for 50 epochs by default.
Checkpoints are saved under `TransGP\Transformer`.

### 3. Evolve and evaluate rules

```powershell
& $python main.py --config .\configs\transgp_train.json
if ($LASTEXITCODE -ne 0) { throw "TransGP evolution failed" }
& $python main.py --config .\configs\transgp_test.json
```

TransGP defaults to a total population of 600 and 50 generations. Evaluation
uses 30 independent replications per task. Results are saved under
`TransGP\MultitaskTransformerGuidedMutationGP\train`.

For other objectives, archive seeds, checkpoint filenames and output formats,
follow the [detailed TransGP guide](TransGP/docs/TRANSGP_USAGE.md).
Running the same scenario and seed again can overwrite its outputs. Only load
archives and checkpoints from trusted sources.

## Unified configuration

Both methods use the same root-level entry points:

- [main.py](main.py): GP/TransGP training, archive generation and evaluation.
- [main_model_train.py](main_model_train.py): Transformer pretraining.

Choose an example from [configs](configs). Its `algorithm` field selects the
workflow; **`MTGP` is basic GP, while `TransGP-archive` generates TransGP's
pretraining data**. Named options override JSON values without changing the file:

```powershell
& $python main.py --config .\configs\transgp_train.json --seed 7
& $python main_model_train.py --config .\configs\transformer_routing.json --epochs 20
```

See the [configuration reference](docs/CONFIGURATION.md) for all eight examples,
supported fields, validation rules and legacy commands. Use `--help` on either
entry point to list its CLI options.

## Documentation

- [Project layout and integration boundaries](docs/PROJECT_LAYOUT.md)
- [Configuration reference and legacy commands](docs/CONFIGURATION.md)
- [Detailed TransGP usage guide](TransGP/docs/TRANSGP_USAGE.md)

## Tests

```powershell
& $python -m unittest discover -s .\TransGP\tests -t . -v
```

Tests use temporary fixtures and reduced-scale experiments to check archive
generation, model training, guided evolution, evaluation, configuration and
basic GP compatibility. No existing archives or checkpoints are required.

## TransGP citation

Meng Xu, Jiao Liu, Hua Yu, and Yew Soon Ong.
**TransGP: Task-Conditioned Transformer-Guided Genetic Programming for Multitask
Dynamic Flexible Job Shop Scheduling.**
*IEEE Transactions on Evolutionary Computation*, Early Access, 2026.
DOI: [10.1109/TEVC.2026.3734959](https://doi.org/10.1109/TEVC.2026.3734959).
