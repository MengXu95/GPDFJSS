from deap import base
from deap import creator
from deap import gp
import MTGPforNewSimulatorBasedOnJava.multi_tree_useDifferentTerminalSet as mt
from MTGPforNewSimulatorBasedOnJava import ea_simple_elitism
from MTGPforNewSimulatorBasedOnJava.ParallelToolbox import ParallelToolbox
from MTGPforNewSimulatorBasedOnJava.selection import *
import sys
from MTGPforNewSimulatorBasedOnJava import saveFile
import time
import random
import numpy as np
import newSimulatorBasedOnJava.jobshop.Objective as Objective
import newSimulatorBasedOnJava.jobshop.SchedulingSet as SchedulingSet
import newSimulatorBasedOnJava.rule.operation.basic.FCFS as FCFS
import newSimulatorBasedOnJava.rule.operation.basic.SPT as SPT
import newSimulatorBasedOnJava.rule.workcenter.basic.WIQ as WIQ
import newSimulatorBasedOnJava.simulation.HeterogeneousSimulation as HeterogeneousSimulation
import newSimulatorBasedOnJava.rule.RuleType as RuleType
import newSimulatorBasedOnJava.rule.operation.evolved.GPRule as GPRule


class shopfloor:
    def __init__(self, **kwargs):
        self.numMachines = 10
        self.numJobs = 5000
        self.warmupJobs = 1000
        self.minNumOperations = 1
        self.maxNumOperations = self.numMachines
        self.dueDateFactor = 1.5
        self.rep = 1
        self.objective = Objective.Objective.TOTAL_TARDINESS
        self.utilLevel = 0.95
        self.schedulingSet = None

        if 'numMachines' in kwargs:
            self.numMachines = kwargs['numMachines']
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


def init_toolbox(toolbox, pset1, pset2):
    REP.init_toolbox(toolbox, pset1, pset2)
    toolbox.register("select", selElitistAndTournament, tournsize=7, elitism=ELITISM)


def init_stats():
    # fitness_stats = tools.Statistics(lambda ind: ind.fitness.values)
    fitness_stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats = tools.MultiStatistics(fitness=fitness_stats)
    stats.register("  avg   ", np.mean)
    stats.register("  std   ", np.std)
    stats.register("  min   ", np.min)
    stats.register("  max   ", np.max)
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



def GPFC_main(dataset_name, seed):
    rd['seed'] = seed
    rd['dataset_name'] = dataset_name
    rd['shop_floor'] = shopfloor()
    num_features = 0 # the initial number of terminals is 0, then I will add more terminals into the pset
    pset1 = gp.PrimitiveSet("MAIN", num_features, prefix="f")
    pset1.context["array"] = np.array
    REP.init_primitives_tree1(pset1)

    pset2 = gp.PrimitiveSet("MAIN", num_features, prefix="f")
    pset2.context["array"] = np.array
    REP.init_primitives_tree2(pset2)

    weights = (-1.,)
    creator.create("FitnessMin", base.Fitness, weights=weights)
    # set up toolbox
    toolbox = ParallelToolbox()  # base.Toolbox()
    init_toolbox(toolbox, pset1, pset2)
    toolbox.register("evaluate", eval_wrapper)

    rd['toolbox'] = toolbox
    pop = toolbox.population(n=POP_SIZE)
    stats = init_stats()
    hof = tools.HallOfFame(1)
    seedRotate = True # added by mengxu 2022.10.13
    # seedRotate = False # added by mengxu 2022.10.13
    pop, logbook, min_fitness, best_ind_all_gen = ea_simple_elitism.eaSimple(pop, toolbox, CXPB, MUTPB, ELITISM, NGEN, seedRotate, rd, stats, halloffame=hof, verbose=True, seed =seed, dataset_name=dataset_name)
    best = hof[0]
    return min_fitness,best, best_ind_all_gen




POP_SIZE =20
NGEN = 5
CXPB = 0.8
MUTPB = 0.2
ELITISM = 10
MAX_HEIGHT = 8
REP = mt  # individual representation {mt (multi-tree) or vt (vector-tree)}
N_TREES = 2
rd = {}

def main(dataset_name, seed):
# if __name__ == "__main__":
#     dataset_name = str(sys.argv[1])
#     seed = int(sys.argv[2])
    random.seed(int(seed))
    np.random.seed(int(seed))
    saveFile.clear_individual_each_gen_to_txt(seed, dataset_name)
    start = time.time()
    min_fitness,p_one,best_ind_all_gen= GPFC_main(dataset_name,seed)
    end = time.time()
    running_time = end - start
    saveFile.save_each_gen_best_individual_meng(seed, dataset_name, best_ind_all_gen)
    saveFile.saveMinFitness(seed, dataset_name, min_fitness)
    saveFile.saveRunningTime(seed, dataset_name, running_time)
    print(min_fitness)
    print("Training time: " + str(running_time))
    print('Training end!')

