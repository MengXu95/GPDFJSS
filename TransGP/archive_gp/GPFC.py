from deap import base
from deap import creator
from deap import gp
import MTGPforNewSimulatorBasedOnJava.multi_tree as mt
from TransGP.archive_gp import ea_simple_elitism
from MTGPforNewSimulatorBasedOnJava.ParallelToolbox import ParallelToolbox
from MTGPforNewSimulatorBasedOnJava.selection import *
from TransGP.archive_gp import saveFile
import os
import time
import random
import numpy as np
import newSimulatorBasedOnJava.jobshop.SchedulingSet as SchedulingSet
import newSimulatorBasedOnJava.simulation.HeterogeneousSimulation as HeterogeneousSimulation
import newSimulatorBasedOnJava.rule.RuleType as RuleType
import newSimulatorBasedOnJava.rule.operation.evolved.GPRule as GPRule
import newSimulatorBasedOnJava.jobshop.Objective as Objective


class shopfloor:
    def __init__(self, **kwargs):
        self.numMachines = 10
        self.numJobs = 124
        self.warmupJobs = 0
        self.minNumOperations = 1
        self.maxNumOperations = self.numMachines
        self.dueDateFactor = 1.5
        self.rep = 1
        self.objective = Objective.Objective.MEAN_FLOWTIME
        self.utilLevel = 0.95
        self.schedulingSet = None

        if 'numMachines' in kwargs:
            self.numMachines = kwargs['numMachines']
            self.maxNumOperations = self.numMachines
        if 'numJobs' in kwargs:
            self.numJobs = kwargs['numJobs']
        if 'warmupJobs' in kwargs:
            self.warmupJobs = kwargs['warmupJobs']
        if 'minNumOperations' in kwargs:
            self.minNumOperations = kwargs['minNumOperations']
        if 'maxNumOperations' in kwargs:
            self.maxNumOperations = kwargs['maxNumOperations']
        if 'dueDateFactor' in kwargs:
            self.dueDateFactor = kwargs['dueDateFactor']
        if 'replication' in kwargs:
            self.rep = kwargs['replication']
        if 'objective' in kwargs:
            self.objective = kwargs['objective']
        if 'utilLevel' in kwargs:
            self.utilLevel = kwargs['utilLevel']

        seed = 968356
        self.schedulingSet = self.createSchedulingSet(seed, self.objective, self.utilLevel)


    def runDynamicSimulation(self, sequencingRule, routingRule):
        objectives = []
        objectives.append(self.objective)
        fitnesses = [-1 for i in range(len(objectives))]
        simulations = self.schedulingSet.getSimulations()
        numRuns = 1
        results = [-1 for i in range(numRuns)]
        startTime = time.time()
        # print(startTime)
        for i in range(numRuns):
            for j in range(len(simulations)):
                # print("run: 0")
                simulation = simulations[j]
                simulation.setSequencingRule(sequencingRule)
                simulation.setRoutingRule(routingRule)
                simulation.run()
                for m in range(len(objectives)):
                    ObjValue = simulation.objectiveValue(objectives[i])
                    fitnesses[i] = ObjValue
                    # if not ObjValue >= np.Infinity:
                    #     fitnesses[i] = ObjValue
                for k in range(1, self.schedulingSet.getReplications()[j]):
                    # print("run: " + str(k))
                    simulation.rerun()
                    ObjValue = simulation.objectiveValue(objectives[i])
                    fitnesses[i] += ObjValue
                    # if not ObjValue >= np.Infinity:
                    #     fitnesses[i] += ObjValue
                simulation.reset()

        for i in range(len(fitnesses)):
            fitnesses[i] /= self.schedulingSet.getReplications()[0]
            # print("objective: " + str(fitnesses[i]))
            # print()
        finishTime = time.time()
        duration = finishTime - startTime
        # print("duration: " + str(duration))
        return fitnesses

    def createSchedulingSet(self, seed, objective, utilLevel):
        trainSimulations = []
        replications = []
        objectives = []
        objectives.append(objective)

        numMachines = self.numMachines
        numJobs = self.numJobs
        warmupJobs = self.warmupJobs
        minNumOperations = self.minNumOperations
        maxNumOperations = self.maxNumOperations
        dueDateFactor = self.dueDateFactor
        rep = self.rep

        simulation = HeterogeneousSimulation.HeterogeneousSimulation(seed, None, None, numMachines, numJobs, warmupJobs,
                                                                     minNumOperations, maxNumOperations, utilLevel,
                                                                     dueDateFactor, False, GPTrain = True)
        trainSimulations.append(simulation)
        replications.append(rep)
        return SchedulingSet.SchedulingSet(trainSimulations, replications, objectives)

    def rotateSeed(self):
        self.schedulingSet.rotateSeed()



def connectedness(cluster):
    print(cluster)


def init_toolbox(toolbox, pset):
    REP.init_toolbox(toolbox, pset)
    toolbox.register("select", selElitistAndTournament, tournsize=TOURSIZE, elitism=ELITISM)

def init_stats():
    def valid_fitness(individual):
        fitness_values = individual.fitness.values
        return fitness_values if all(np.isfinite(fitness_values)) else None

    fitness_stats = tools.Statistics(valid_fitness)
    stats = tools.MultiStatistics(fitness=fitness_stats)
    stats.register("  avg   ", lambda data: np.mean([f for f in data if f is not None]))
    stats.register("  std   ", lambda data: np.std([f for f in data if f is not None]))
    stats.register("  min   ", lambda data: np.min([f for f in data if f is not None]))
    stats.register("  max   ", lambda data: np.max([f for f in data if f is not None]))
    return stats

def evaluate(individual, toolbox):
    # create the shop floor instance
    spf = rd['shop_floor']
    sequencingRule = GPRule.GPRule(RuleType.RuleType.SEQUENCING, individual[0])
    routingRule = GPRule.GPRule(RuleType.RuleType.ROUTING, individual[1])
    # currently, this is only for single objective DFJSS by mengxu 2022.11.17
    fitnesses = spf.runDynamicSimulation(sequencingRule, routingRule)
    scores = [fitnesses[0]]
    return scores


def eval_wrapper(*args, **kwargs):
    return evaluate(*args, **kwargs, toolbox=rd['toolbox'])
    # return evaluate(*args, **kwargs, toolbox=rd['toolbox'], data=rd['data'], labels=rd['labels'])


# copies data over from parent process
def init_data(rundata):
    global rd
    rd = rundata



UTIL_MACHINE_MAP = {
    0.75: 6,
    0.85: 8,
    0.95: 10,
}


def machine_number_for_util(utilLevel, fallback=10):
    return UTIL_MACHINE_MAP.get(round(float(utilLevel), 2), fallback)


def GPFC_main(dataset_name, seed, objective, utilLevel, machine_num=None, wall_clock_limit_seconds=None, start_time=None):
    rd['seed'] = seed
    rd['dataset_name'] = dataset_name
    if machine_num is None:
        machine_num = machine_number_for_util(utilLevel)
    rd['shop_floor'] = shopfloor(objective=objective, utilLevel=utilLevel, numMachines=machine_num)
    num_features = 0 # the initial number of terminals is 0, then I will add more terminals into the pset
    pset = gp.PrimitiveSet("MAIN", num_features, prefix="f")
    pset.context["array"] = np.array
    REP.init_primitives(pset)
    weights = (-1.,)
    creator.create("FitnessMin", base.Fitness, weights=weights)
    # set up toolbox
    toolbox = ParallelToolbox()  # base.Toolbox()
    init_toolbox(toolbox, pset)
    toolbox.register("evaluate", eval_wrapper)

    rd['toolbox'] = toolbox
    pop = toolbox.population(n=POP_SIZE)
    stats = init_stats()
    hof = tools.HallOfFame(1)
    seedRotate = True # added by mengxu 2022.10.13
    # seedRotate = False # added by mengxu 2022.10.13
    pop, logbook, min_fitness, best_ind_all_gen = ea_simple_elitism.eaSimple(pop, toolbox, CXPB, MUTPB, REPPB, ELITISM, NGEN, seedRotate, rd, stats, halloffame=hof, verbose=True, seed =seed, dataset_name=dataset_name, wall_clock_limit_seconds=wall_clock_limit_seconds, start_time=start_time)
    best = hof[0]
    return min_fitness,best, best_ind_all_gen




def env_int(name, default):
    value = os.environ.get(name)
    return default if value is None else int(value)


POP_SIZE = env_int("MTGP_POP_SIZE", 200)
NGEN = 50
CXPB = 0.8
MUTPB = 0.15
REPPB = 0.05
TOURSIZE = env_int("MTGP_TOURSIZE", 4)
ELITISM = 4
MAX_HEIGHT = 6
REP = mt  # individual representation {mt (multi-tree) or vt (vector-tree)}
N_TREES = 2
rd = {}

def main(dataset_name, seed, objective, utilLevel, machine_num=None, wall_clock_limit_seconds=None):
# if __name__ == "__main__":
#     dataset_name = str(sys.argv[1])
#     seed = int(sys.argv[2])
    random.seed(int(seed))
    np.random.seed(int(seed))
    saveFile.clear_individual_each_gen_to_txt(seed, dataset_name)
    start = time.time()
    min_fitness,p_one,best_ind_all_gen= GPFC_main(dataset_name,seed,objective,utilLevel, machine_num=machine_num, wall_clock_limit_seconds=wall_clock_limit_seconds, start_time=start)
    end = time.time()
    running_time = end - start
    saveFile.save_each_gen_best_individual_meng(seed, dataset_name, best_ind_all_gen)
    saveFile.saveMinFitness(seed, dataset_name, min_fitness)
    saveFile.saveRunningTime(seed, dataset_name, running_time)
    print(min_fitness)
    print("Training time: " + str(running_time))
    print('Training end!')
