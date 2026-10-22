# Configuration Reference

For installation and the recommended execution order, start with the
[README](../README.md).

## Choose a workflow

Run commands from the repository root. The examples use PowerShell:

```powershell
$python = (Resolve-Path .\.venv\Scripts\python.exe).Path
```

| Configuration | Entry point | `algorithm` / stage |
| --- | --- | --- |
| [gp_train.json](../configs/gp_train.json) | `main.py` | `MTGP`: basic GP training |
| [gp_test.json](../configs/gp_test.json) | `main.py` | `MTGP-test`: basic GP evaluation |
| [transgp_archive.json](../configs/transgp_archive.json) | `main.py` | `TransGP-archive`: generate a single-task pretraining archive |
| [transgp_archive_test.json](../configs/transgp_archive_test.json) | `main.py` | `TransGP-archive-test`: evaluate an archive-GP rule |
| [transgp_train.json](../configs/transgp_train.json) | `main.py` | `TransGP`: guided multitask evolution |
| [transgp_test.json](../configs/transgp_test.json) | `main.py` | `TransGP-test`: independent evaluation |
| [transformer_sequencing.json](../configs/transformer_sequencing.json) | `main_model_train.py` | `TransGP` / sequencing model pretraining |
| [transformer_routing.json](../configs/transformer_routing.json) | `main_model_train.py` | `TransGP` / routing model pretraining |

Use the matching entry point for each configuration. For example, the TransGP
evolution configuration is:

```json
{
  "algorithm": "TransGP",
  "dataset_name": "Fmax-0.75-0.85-0.95-1.5",
  "objective": "Fmax",
  "seed": 0
}
```

After generating the required archives and both checkpoints:

```powershell
& $python main.py --config .\configs\transgp_train.json
```

## Experiment fields

For [main.py](../main.py), `algorithm`, `dataset_name` and `objective` are
required. `seed` defaults to 0 and must be a nonnegative integer.

Scenario names use `<objective>-<utilization...>-1.5`, for example
`Fmax-0.85-1.5` for a single task or `Fmax-0.75-0.85-0.95-1.5` for TransGP.
The trailing `1.5` is the due-date factor, not utilization.

`objective` accepts short identifiers (`Fmax`, `Fmean`, `WFmean`, `Tmax`,
`Tmean`, `WTmean`, `makespan`) or simulator names such as `max-flowtime`;
it must agree with the scenario name.

| Optional field | Supported use |
| --- | --- |
| `utilization` | Single-task runs only; if provided, must match `dataset_name`. Otherwise inferred from it |
| `machine_num` | `TransGP-archive` only; integer >= 2. Defaults to 6/8/10 at utilization 0.75/0.85/0.95 |
| `wall_clock_seconds` | `TransGP-archive` only; finite positive generation-boundary budget |
| `skip_existing` | Boolean for `TransGP-test` or `TransGP-archive-test`; defaults to false |

`MTGP` retains the basic GP settings: ten machines, population 50 and
5 generations. `TransGP-archive` uses population 200 and 50 generations to
generate the archives required for pretraining. They are not interchangeable.

## Model-training fields

For [main_model_train.py](../main_model_train.py):

| Field | Values |
| --- | --- |
| `dataset_name` | Required combined TransGP scenario |
| `rule_type` | Required: `sequencing` or `routing` |
| `algorithm` | `TransGP` (default) |
| `model_type` | `Transformer` (default) |
| `epochs` | Positive integer, default 50 |

Default archive seed mappings are available for **Fmax** (0, 1, 2),
**Fmean** (3, 4, 5) and **Tmean** (6, 7, 8). Each objective needs its own
archives and both checkpoints. See the
[TransGP guide](../TransGP/docs/TRANSGP_USAGE.md#4-generate-gp-archives).

Basic GP does not need model pretraining. CVAE and LSTM are not included.

## Override configuration values

Named CLI options override JSON values without modifying the file:

```powershell
& $python main.py --config .\configs\transgp_train.json --seed 7
& $python main_model_train.py --config .\configs\transformer_routing.json --epochs 20

# Change both scenario and objective when switching objectives.
& $python main.py --config .\configs\transgp_archive.json --dataset-name Fmean-0.75-1.5 --objective Fmean --seed 3

& $python main.py --help
& $python main_model_train.py --help
```

Configurations select workflows and entry-point arguments; population and
generation defaults remain in the algorithm implementation modules.

## Validation and paths

- Unknown fields, invalid types, duplicate keys and incompatible task settings
  fail explicitly rather than being ignored.
- Configuration files are read relative to the caller's working directory
  **before** execution changes directory.
- Basic GP writes under the repository root; all TransGP stages write under
  `TransGP`. The caller's working directory is restored on completion or failure.
- Running the same scenario and seed again can overwrite existing outputs.

See the [output and environment-variable reference](../TransGP/docs/TRANSGP_USAGE.md#8-outputs-and-configuration)
for custom TransGP output paths.

## Legacy command compatibility

Existing positional commands remain supported through the unified root
implementations:

```powershell
# Original basic GP: DATASET OBJECTIVE UTILIZATION SEED ALGO.
& $python main.py Fmax-0.85-1.5 max-flowtime 0.85 0 MTGP

# Former TransGP archive command: MTGP here maps to TransGP-archive.
& $python -m TransGP.main Fmax-0.75-1.5 Fmax 0 MTGP

# Both model-training commands reach the same root implementation.
& $python main_model_train.py Fmax-0.75-0.85-0.95-1.5 routing Transformer 50
& $python -m TransGP.main_model_train Fmax-0.75-0.85-0.95-1.5 routing Transformer 50

# Former evaluation command also delegates to main.py.
& $python -m TransGP.main_GP_TransGP_test TransGP Fmax-0.75-0.85-0.95-1.5 Fmax 0
```

Do not mix positional syntax with `--config` or the new named options.
Use the JSON configurations for new experiments.
