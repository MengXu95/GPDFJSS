"""Validated JSON configuration for the shared experiment entry points."""

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path

from TransGP.experiment_config import parse_scenario, require_transgp_tasks


PROJECT_ROOT = Path(__file__).resolve().parent
TRANSGP_ROOT = PROJECT_ROOT / "TransGP"
ALGORITHMS = (
    "MTGP", "MTGP-test", "TransGP-archive", "TransGP-archive-test",
    "TransGP", "TransGP-test",
)
OBJECTIVE_ALIASES = {
    "max-flowtime": "Fmax",
    "mean-flowtime": "Fmean",
    "mean-weighted-flowtime": "WFmean",
    "max-tardiness": "Tmax",
    "mean-tardiness": "Tmean",
    "mean-weighted-tardiness": "WTmean",
}


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is not supported: {value}")


def load_configuration(path: str | Path) -> dict[str, object]:
    with Path(path).open(encoding="utf-8-sig") as handle:
        data = json.load(handle, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object")
    return data


def _check_fields(data: Mapping[str, object], allowed: set[str]) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError("Unsupported configuration fields: " + ", ".join(sorted(unknown)))


def _string(data: Mapping[str, object], name: str, default: str | None = None) -> str:
    value = data.get(name, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _integer(data: Mapping[str, object], name: str, default: int, minimum: int) -> int:
    value = data.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(data: Mapping[str, object], name: str) -> float | None:
    if name not in data:
        return None
    value = data[name]
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class ExperimentConfig:
    algorithm: str
    dataset_name: str
    objective: str
    seed: int
    utilizations: tuple[float, ...]
    machine_num: int | None
    wall_clock_seconds: float | None
    skip_existing: bool

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "ExperimentConfig":
        _check_fields(data, {
            "algorithm", "dataset_name", "objective", "seed", "utilization",
            "machine_num", "wall_clock_seconds", "skip_existing",
        })
        algorithm = _string(data, "algorithm")
        if algorithm not in ALGORITHMS:
            raise ValueError("algorithm must be one of: " + ", ".join(ALGORITHMS))
        dataset_name = _string(data, "dataset_name")
        objective = _string(data, "objective")
        objective = OBJECTIVE_ALIASES.get(objective, objective)
        scenario = parse_scenario(dataset_name, objective)
        seed = _integer(data, "seed", 0, 0)
        if algorithm in ("TransGP", "TransGP-test"):
            require_transgp_tasks(scenario)
        elif algorithm != "TransGP-archive-test" and len(scenario.utilizations) != 1:
            raise ValueError(f"{algorithm} requires a single-task scenario")
        utilization = _number(data, "utilization")
        if "utilization" in data and (
            algorithm not in ("MTGP", "MTGP-test", "TransGP-archive")
            or scenario.utilizations != (utilization,)
        ):
            raise ValueError("utilization is single-task only and must match dataset_name")
        machine_num = None
        if "machine_num" in data:
            if algorithm != "TransGP-archive":
                raise ValueError("machine_num is supported only for TransGP-archive")
            machine_num = _integer(data, "machine_num", 10, 2)
        wall_clock_seconds = _number(data, "wall_clock_seconds")
        if "wall_clock_seconds" in data and (
            algorithm != "TransGP-archive"
            or wall_clock_seconds is None or wall_clock_seconds <= 0
        ):
            raise ValueError("wall_clock_seconds must be positive and is supported only for TransGP-archive")
        skip_existing = data.get("skip_existing", False)
        if not isinstance(skip_existing, bool):
            raise ValueError("skip_existing must be a boolean")
        if "skip_existing" in data and algorithm not in ("TransGP-test", "TransGP-archive-test"):
            raise ValueError("skip_existing is supported only for TransGP saved-rule evaluation")
        return cls(
            algorithm, dataset_name, objective, seed, scenario.utilizations,
            machine_num, wall_clock_seconds, skip_existing,
        )


@dataclass(frozen=True)
class ModelTrainingConfig:
    dataset_name: str
    rule_type: str
    epochs: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "ModelTrainingConfig":
        _check_fields(data, {"algorithm", "dataset_name", "rule_type", "model_type", "epochs"})
        if _string(data, "algorithm", "TransGP") != "TransGP":
            raise ValueError("Model pretraining is supported only for algorithm TransGP")
        if _string(data, "model_type", "Transformer") != "Transformer":
            raise ValueError("Only the Transformer model is included in this project")
        dataset_name = _string(data, "dataset_name")
        scenario = parse_scenario(dataset_name)
        require_transgp_tasks(scenario)
        if scenario.objective not in ("Fmax", "Fmean", "Tmean"):
            raise ValueError("Default archive seeds are defined only for Fmax, Fmean and Tmean")
        rule_type = _string(data, "rule_type")
        if rule_type not in ("sequencing", "routing"):
            raise ValueError("rule_type must be sequencing or routing")
        return cls(dataset_name, rule_type, _integer(data, "epochs", 50, 1))
