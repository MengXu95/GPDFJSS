from deap import base
from deap import creator
from deap import gp
import TransGP.MultitaskTransformerGuidedMutationGP.multi_tree as mt
from TransGP.MultitaskTransformerGuidedMutationGP import ea_simple_elitism
from MTGPforNewSimulatorBasedOnJava.ParallelToolbox import ParallelToolbox
from TransGP.MultitaskTransformerGuidedMutationGP.selection import *
from deap import tools
from TransGP.MultitaskTransformerGuidedMutationGP import saveFile
import time
import random
import os
import numpy as np
import newSimulatorBasedOnJava.jobshop.Objective as Objective
import newSimulatorBasedOnJava.jobshop.SchedulingSet as SchedulingSet
import newSimulatorBasedOnJava.simulation.HeterogeneousSimulation as HeterogeneousSimulation
import newSimulatorBasedOnJava.rule.RuleType as RuleType
import newSimulatorBasedOnJava.rule.operation.evolved.GPRule as GPRule


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
    spf = rd['shop_floor'][individual.task_tag]
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



def GPFC_main(dataset_name, seed, objective, utilLevel):
    rd['seed'] = seed
    rd['dataset_name'] = dataset_name
    rd['shop_floor'] = []
    machine_number = 10
    for each_uti in utilLevel:
        if each_uti == 0.75:
            machine_number = 6
        elif each_uti == 0.85:
            machine_number = 8
        elif each_uti == 0.95:
            machine_number = 10
        else:
            print("Scenario wrong!")
        task = shopfloor(objective=objective, utilLevel=each_uti, numMachines=machine_number)
        rd['shop_floor'].append(task)

    # the following is original
    # rd['shop_floor'] = shopfloor(objective=objective, utilLevel=utilLevel)
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
    # add by mengxu for random assign task_tag to each individual
    task_type_count = [0 for _ in utilLevel]
    for ind in pop:
        task_tag = np.random.randint(len(utilLevel))
        ind.task_tag = task_tag
        task_type_count[task_tag] += 1

    print("Task count: ", task_type_count)
    stats = init_stats()
    hof = tools.HallOfFame(1)
    seedRotate = True # added by mengxu 2022.10.13
    # seedRotate = False # added by mengxu 2022.10.13
    pop, logbook, min_fitness, best_ind_all_gen, top_inds_final_gen = ea_simple_elitism.eaSimple(pop, toolbox, CXPB, MUTPB, REPPB, ELITISM, NGEN, seedRotate, rd, stats, halloffame=hof, verbose=True, seed =seed, dataset_name=dataset_name)
    best = hof[0]
    return min_fitness,best, best_ind_all_gen, top_inds_final_gen




def env_float(name, default):
    value = os.environ.get(name)
    return default if value is None else float(value)


POP_SIZE = 600
NGEN = 50
CXPB = env_float("MTGP_CXPB", 0.8)
MUTPB = env_float("MTGP_MUTPB", 0.15)
REPPB = env_float("MTGP_REPPB", 0.05)
TOURSIZE = 4
ELITISM = 4
MAX_HEIGHT = 6
REP = mt  # individual representation {mt (multi-tree) or vt (vector-tree)}
N_TREES = 2
rd = {}

def main(dataset_name, seed, objective, utilLevel):
# if __name__ == "__main__":
#     dataset_name = str(sys.argv[1])
#     seed = int(sys.argv[2])
    random.seed(int(seed))
    np.random.seed(int(seed))

    dataset_name_parts = dataset_name.split('-')
    num_task =len(utilLevel)
    for task_id in range(num_task):
        dataset_name_eachtask = dataset_name_parts[0] + '-' + str(utilLevel[task_id]) + '-' + dataset_name_parts[-1]
        saveFile.clear_individual_each_gen_to_txt(seed, dataset_name_eachtask)
    start = time.time()
    min_fitness,p_one,best_ind_all_gen,top_inds_final_gen= GPFC_main(dataset_name,seed, objective, utilLevel)
    end = time.time()
    running_time = end - start
    for task_id in range(num_task):
        dataset_name_eachtask = dataset_name_parts[0] + '-' + str(utilLevel[task_id]) + '-' + dataset_name_parts[-1]
        saveFile.save_each_gen_best_individual_meng(seed, dataset_name_eachtask, best_ind_all_gen[task_id])
        saveFile.save_top_inds_final_gen_meng(seed, dataset_name_eachtask, top_inds_final_gen[task_id])
        saveFile.saveMinFitness(seed, dataset_name_eachtask, min_fitness[task_id])
        saveFile.saveRunningTime(seed, dataset_name_eachtask, running_time)
    print(min_fitness)
    print("Training time: " + str(running_time))
    print('Training end!')
