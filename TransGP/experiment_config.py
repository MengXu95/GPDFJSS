"""Shared scenario validation for the TransGP release entry points."""

from dataclasses import dataclass
import math
from pathlib import Path


OBJECTIVES = ("Fmax", "Fmean", "WFmean", "Tmax", "Tmean", "WTmean", "makespan")
TASK_UTILIZATIONS = (0.75, 0.85, 0.95)
PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Scenario:
    objective: str
    utilizations: tuple[float, ...]


def parse_scenario(dataset_name: str, objective: str | None = None) -> Scenario:
    parts = dataset_name.split("-")
    if len(parts) < 3 or parts[0] not in OBJECTIVES:
        raise ValueError("Use a scenario such as Fmax-0.75-1.5 or Fmax-0.75-0.85-0.95-1.5")
    try:
        utilizations = tuple(float(value) for value in parts[1:-1])
        due_date_factor = float(parts[-1])
    except ValueError as error:
        raise ValueError("Scenario utilization levels and due-date factor must be numeric") from error
    if any(not math.isfinite(value) or not 0 < value < 1 for value in utilizations):
        raise ValueError("Utilization levels must be finite numbers strictly between 0 and 1")
    if due_date_factor != 1.5:
        raise ValueError("The release simulator uses due-date factor 1.5")
    if objective is not None and parts[0] != objective:
        raise ValueError("Scenario objective must match the objective argument")
    canonical = "-".join((parts[0], *(str(value) for value in utilizations), "1.5"))
    if dataset_name != canonical:
        raise ValueError(f"Use the canonical scenario name {canonical} to match saved artifact paths")
    return Scenario(parts[0], utilizations)


def require_transgp_tasks(scenario: Scenario) -> None:
    if scenario.utilizations != TASK_UTILIZATIONS:
        raise ValueError("TransGP requires the ordered tasks 0.75, 0.85, 0.95 (6, 8, 10 machines)")


def checkpoint_paths(dataset_name: str) -> tuple[Path, Path]:
    root = PROJECT_ROOT / "Transformer"
    filename = f"{dataset_name}_diffusion_transformer_original.pt"
    return (
        root / "sequencing_transformer_multitask" / filename,
        root / "routing_transformer_multitask" / filename,
    )


def resolve_objective(objective: str) -> str:
    from newSimulatorBasedOnJava.jobshop.Objective import Objective

    return {
        "Fmax": Objective.MAX_FLOWTIME,
        "Fmean": Objective.MEAN_FLOWTIME,
        "WFmean": Objective.MEAN_WEIGHTED_FLOWTIME,
        "Tmax": Objective.MAX_TARDINESS,
        "Tmean": Objective.MEAN_TARDINESS,
        "WTmean": Objective.MEAN_WEIGHTED_TARDINESS,
        "makespan": Objective.MAKESPAN,
    }[objective]
