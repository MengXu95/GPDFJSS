import numpy as np
import itertools
import pandas as pd
import random
import math
from statistics import mean, stdev
import time
from deap import gp
from deap.gp import Primitive, Terminal, PrimitiveTree
import newSimulatorBasedOnJava.jobshop.Objective as Objective
import newSimulatorBasedOnJava.jobshop.SchedulingSet as SchedulingSet
import newSimulatorBasedOnJava.simulation.HeterogeneousSimulation as HeterogeneousSimulation
import newSimulatorBasedOnJava.rule.RuleType as RuleType
import newSimulatorBasedOnJava.rule.operation.evolved.GPRule as GPRule

import MTGPforNewSimulatorBasedOnJava.LoadIndividual as diffGPload
import os

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

        seed = 77777
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

def evaluate(individual, spf):
    sequencingRule = GPRule.GPRule(RuleType.RuleType.SEQUENCING, individual[0])
    routingRule = GPRule.GPRule(RuleType.RuleType.ROUTING, individual[1])
    # currently, this is only for single objective DFJSS by mengxu 2022.11.17
    fitnesses = spf.runDynamicSimulation(sequencingRule, routingRule)
    scores = [fitnesses[0]]
    return scores


def protected_div(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x


def init_primitives(pset):
    # add function
    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(np.maximum, 2)
    pset.addPrimitive(np.minimum, 2)
    # terminals for sequencing and routing in my paper
    pset.addTerminal(str('NIQ'))  # add by mengxu
    pset.addTerminal(str('WIQ'))  # add by mengxu
    pset.addTerminal(str('MWT'))  # add by mengxu
    pset.addTerminal(str('PT'))  # add by mengxu
    pset.addTerminal(str('NPT'))  # add by mengxu
    pset.addTerminal(str('OWT'))  # add by mengxu
    pset.addTerminal(str('WKR'))  # add by mengxu
    pset.addTerminal(str('NOR'))  # add by mengxu
    # pset.addTerminal('W')  # add by mengxu
    pset.addTerminal(str('TIS'))  # add by mengxu
    # pset.addTerminal('TRANT')  # add by mengxu
    # adviced terminal by Yi 2022.10.31
    pset.addTerminal(str('SLACK'))  # add by mengxu


def main(algo, dataset_name, seed, objective, utilLevel, machine_num=10):
    randomSeed = 777
    random.seed(int(randomSeed))
    np.random.seed(int(randomSeed))

    num_features = 0  # the initial number of terminals is 0, then I will add more terminals into the pset
    pset = gp.PrimitiveSet("MAIN", num_features, prefix="f")
    pset.context["array"] = np.array
    init_primitives(pset)

    # create the shop floor instance
    spf = shopfloor(objective=objective, utilLevel=utilLevel, replication=30, machine_num=machine_num)

    dict_best_individuals = diffGPload.load_individual_from_gen(seed, dataset_name)
    training_time = diffGPload.load_training_time(seed, dataset_name)
    training_fitness = diffGPload.load_min_fitness(seed, dataset_name)
    test_fitness = []
    individual_size = []

    for idx in range(len(dict_best_individuals)):
        print("Generation: " + str(idx))
        ind= dict_best_individuals.get(idx)
        individual = []

        # Convert string list to a string expression DEAP can parse
        expr_sequencing_str = " ".join(ind[0])
        sequencing_tree = PrimitiveTree.from_string(expr_sequencing_str, pset)
        # Convert string list to a string expression DEAP can parse
        expr_routing_str = " ".join(ind[1])
        routing_tree = PrimitiveTree.from_string(expr_routing_str, pset)

        individual.append(sequencing_tree)
        individual.append(routing_tree)

        test_fit = evaluate(individual, spf)
        test_fitness.append(test_fit[0])
        ind_size = len(individual[0]) + len(individual[1])
        individual_size.append(ind_size)

    df = pd.DataFrame({
        'algo': [algo for _ in range(len(dict_best_individuals))],
        'run': [seed for _ in range(len(dict_best_individuals))],
        'train_fitness': training_fitness,
        'training_time': training_time,
        'test_fitness': test_fitness,
        'individual_size': individual_size
    })

    # Define the folder path
    folder_path = f"./{algo}/test"
    # Check if the folder exists
    if not os.path.exists(folder_path):
        # Create the folder
        os.makedirs(folder_path)
    # Define the folder path
    folder_path = f"./{algo}/test/scenario_{dataset_name}"
    # Check if the folder exists
    if not os.path.exists(folder_path):
        # Create the folder
        os.makedirs(folder_path)

    df.to_csv(folder_path + f"/{seed}_{dataset_name}_test.csv", index=False)
    print('Test end!')