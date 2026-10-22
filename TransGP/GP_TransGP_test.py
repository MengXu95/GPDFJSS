import pickle
import time
import os
from pathlib import Path

import numpy as np
from deap import gp
from deap.gp import PrimitiveTree

import MTGPforNewSimulatorBasedOnJava.multi_tree as mt
import newSimulatorBasedOnJava.jobshop.Objective as Objective
import newSimulatorBasedOnJava.jobshop.SchedulingSet as SchedulingSet
import newSimulatorBasedOnJava.rule.RuleType as RuleType
import newSimulatorBasedOnJava.rule.operation.evolved.GPRule as GPRule
import newSimulatorBasedOnJava.simulation.HeterogeneousSimulation as HeterogeneousSimulation

METHOD_ROOTS = {
    "GP": Path("archive_gp"),
    "TransGP": Path(os.environ.get("TRANSGP_METHOD_ROOT", "MultitaskTransformerGuidedMutationGP")),
}
UTIL_MACHINE_MAP = {
    0.75: 6,
    0.85: 8,
    0.95: 10,
}
OBJECTIVE_MAP = {
    "Fmax": Objective.Objective.MAX_FLOWTIME,
    "Fmean": Objective.Objective.MEAN_FLOWTIME,
    "WFmean": Objective.Objective.MEAN_WEIGHTED_FLOWTIME,
    "Tmax": Objective.Objective.MAX_TARDINESS,
    "Tmean": Objective.Objective.MEAN_TARDINESS,
    "WTmean": Objective.Objective.MEAN_WEIGHTED_TARDINESS,
    "makespan": Objective.Objective.MAKESPAN,
    "max-flowtime": Objective.Objective.MAX_FLOWTIME,
    "mean-flowtime": Objective.Objective.MEAN_FLOWTIME,
    "mean-weighted-flowtime": Objective.Objective.MEAN_WEIGHTED_FLOWTIME,
    "max-tardiness": Objective.Objective.MAX_TARDINESS,
    "mean-tardiness": Objective.Objective.MEAN_TARDINESS,
    "mean-weighted-tardiness": Objective.Objective.MEAN_WEIGHTED_TARDINESS,
}


def resolve_objective(objective):
    return OBJECTIVE_MAP.get(objective, objective)


def machine_number_for_util(util_level, fallback=10):
    return UTIL_MACHINE_MAP.get(round(float(util_level), 2), fallback)


def scenario_for_task(dataset_name, util_level):
    parts = dataset_name.split("-")
    if len(parts) >= 3:
        return f"{parts[0]}-{util_level}-{parts[-1]}"
    return dataset_name


def build_pset():
    pset = gp.PrimitiveSet("MAIN", 0, prefix="f")
    pset.context["array"] = np.array
    mt.init_primitives(pset)
    return pset


def tree_from_tokens(tokens, pset):
    return PrimitiveTree.from_string(" ".join(tokens), pset)


def candidate_files(method, task_name, seed):
    base = METHOD_ROOTS[method] / "train" / f"scenario_{task_name}"
    return [
        base / f"{seed}_meng_top_individuals_final_gen_{task_name}.pkl",
        base / f"{seed}_meng_individual_{task_name}.pkl",
        base / f"{seed}_meng_archive_{task_name}.pkl",
    ]


def load_saved_individual(method, task_name, seed, pset):
    source_file = None
    for path in candidate_files(method, task_name, seed):
        if path.exists():
            source_file = path
            break
    if source_file is None:
        searched = ", ".join(str(path) for path in candidate_files(method, task_name, seed))
        raise FileNotFoundError(f"Missing saved {method} individual. Looked for: {searched}")
    with open(source_file, "rb") as handle:
        individuals = pickle.load(handle)
    if not individuals:
        raise ValueError(f"Empty saved individual file: {source_file}")
    selected_key = min(individuals.keys()) if "top_individuals_final_gen" in source_file.name else max(individuals.keys())
    individual = individuals[selected_key]
    return [tree_from_tokens(individual[0], pset), tree_from_tokens(individual[1], pset)], source_file, selected_key


class TestShopFloor:
    def __init__(self, objective, util_level, num_machines, replications=30):
        self.objective = objective
        self.util_level = util_level
        self.num_machines = num_machines
        self.num_jobs = 124
        self.warmup_jobs = 0
        self.min_num_operations = 1
        self.max_num_operations = num_machines
        self.due_date_factor = 1.5
        self.replications = replications
        self.scheduling_set = self.create_scheduling_set(77777)

    def create_scheduling_set(self, seed):
        simulation = HeterogeneousSimulation.HeterogeneousSimulation(
            seed,
            None,
            None,
            self.num_machines,
            self.num_jobs,
            self.warmup_jobs,
            self.min_num_operations,
            self.max_num_operations,
            self.util_level,
            self.due_date_factor,
            False,
            GPTrain=True,
        )
        return SchedulingSet.SchedulingSet([simulation], [self.replications], [self.objective])

    def evaluate(self, individual):
        sequencing_rule = GPRule.GPRule(RuleType.RuleType.SEQUENCING, individual[0])
        routing_rule = GPRule.GPRule(RuleType.RuleType.ROUTING, individual[1])
        simulations = self.scheduling_set.getSimulations()
        fitness = 0.0
        for simulation_index, simulation in enumerate(simulations):
            simulation.setSequencingRule(sequencing_rule)
            simulation.setRoutingRule(routing_rule)
            simulation.run()
            fitness += simulation.objectiveValue(self.objective)
            for _ in range(1, self.scheduling_set.getReplications()[simulation_index]):
                simulation.rerun()
                fitness += simulation.objectiveValue(self.objective)
            simulation.reset()
        return fitness / self.scheduling_set.getReplications()[0]


def output_paths(method, dataset_name, seed):
    train_dir = METHOD_ROOTS[method] / "train" / f"scenario_{dataset_name}"
    train_dir.mkdir(parents=True, exist_ok=True)
    return (
        train_dir / f"{seed}_test_fitness{dataset_name}.npy",
        train_dir / f"{seed}_test_running_time{dataset_name}.npy",
    )


def test(method, dataset_name, objective, seed, util_levels, skip_existing=False):
    objective_value = resolve_objective(objective)
    if not isinstance(util_levels, list):
        util_levels = [util_levels]
    fitness_path, running_time_path = output_paths(method, dataset_name, seed)
    if skip_existing and fitness_path.exists():
        values = np.asarray(np.load(fitness_path), dtype=float).reshape(-1)
        if len(values) == len(util_levels) and np.isfinite(values).all():
            print(f"Skip {method} seed={seed} scenario={dataset_name}: {fitness_path} exists", flush=True)
            return values
    pset = build_pset()
    start = time.time()
    fitnesses = []
    for util_level in util_levels:
        task_name = scenario_for_task(dataset_name, util_level)
        machine_num = machine_number_for_util(util_level)
        individual, source_file, selected_key = load_saved_individual(method, task_name, seed, pset)
        shop_floor = TestShopFloor(objective_value, float(util_level), machine_num)
        fitness = shop_floor.evaluate(individual)
        fitnesses.append(fitness)
        print(
            f"{method} {task_name} machines={machine_num} fitness={fitness} "
            f"source={source_file.name} key={selected_key}",
            flush=True,
        )
    running_time = time.time() - start
    np.save(fitness_path, np.array(fitnesses, dtype=float))
    np.save(running_time_path, np.array(running_time, dtype=float))
    print(f"{method} test running time: {running_time}", flush=True)
    return np.array(fitnesses, dtype=float)
