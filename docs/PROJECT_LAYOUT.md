# Project Layout and Integration Boundaries

For installation and runnable workflows, start with the [README](../README.md).

## Entry points and configuration

| Path | Purpose |
| --- | --- |
| [main.py](../main.py) | Unified JSON-configured GP/TransGP training and evaluation CLI |
| [main_model_train.py](../main_model_train.py) | Unified Transformer model-pretraining CLI |
| [configs](../configs) | Ready-to-run configurations for each workflow stage |
| [experiment_config.py](../experiment_config.py) | Shared configuration loading and validation |

The root-level entry points handle both approaches. The former TransGP entry
modules are compatibility adapters, not separate implementations. See the
[configuration reference](CONFIGURATION.md) for supported algorithms and commands.

## Shared and basic GP components

| Path | Purpose |
| --- | --- |
| [MTGPforNewSimulatorBasedOnJava](../MTGPforNewSimulatorBasedOnJava) | Basic GP implementation; shared primitives, operators, toolbox and archive loader |
| [newSimulatorBasedOnJava](../newSimulatorBasedOnJava) | Shared Python DFJSS simulator |
| [Test](../Test) | Basic GP evaluation |
| [Utils](../Utils) | Shared rule-size and fitness-diversity utilities |
| [paintingAnalysis](../paintingAnalysis), [wilcoxonTest](../wilcoxonTest) | Plotting and statistical analysis tools |

## TransGP-specific components

| Path | Purpose |
| --- | --- |
| [TransGP/archive_gp](../TransGP/archive_gp) | GP configuration, evolution and persistence used to generate pretraining archives |
| [TransGP/Transformer](../TransGP/Transformer) | Task-conditioned sequencing and routing tree generators |
| [TransGP/MultitaskTransformerGuidedMutationGP](../TransGP/MultitaskTransformerGuidedMutationGP) | Task-tagged population, guided mutation, evolution and saved rules |
| [TransGP/GP_TransGP_test.py](../TransGP/GP_TransGP_test.py) | Independent evaluation of saved archive-GP and TransGP rules |
| [TransGP/experiment_config.py](../TransGP/experiment_config.py) | Scenario validation, task constraints and checkpoint paths |
| [TransGP/tests](../TransGP/tests) | Configuration, integration and reduced-scale workflow tests |
| [TransGP/docs/TRANSGP_USAGE.md](../TransGP/docs/TRANSGP_USAGE.md) | Detailed workflow, parameters and output formats |

## Integration boundaries

TransGP was imported from the `main` branch of
[MengXu95/TransGP](https://github.com/MengXu95/TransGP), commit
`bd01df0f81d635ed4c194a59801243a4ce8c86c0`. Only its single-objective workflow
is included, without other methods, experiment results, archives, pretrained
weights or virtual environments.

- The simulator and compatible GP helpers are reused rather than copied.
- TransGP-specific imports are namespaced under `TransGP`.
- The archive generator retains its own evolution settings without changing
  the basic GP defaults.
- Basic GP outputs remain under the repository root; TransGP outputs remain
  inside `TransGP`.

These boundaries keep the original basic GP behavior separate from
TransGP-specific training and evaluation.
