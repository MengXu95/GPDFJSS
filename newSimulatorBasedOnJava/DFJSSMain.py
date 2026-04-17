import time
from multiprocessing import cpu_count, Pool

import numpy as np

import newSimulatorBasedOnJava.jobshop.Objective as Objective
import newSimulatorBasedOnJava.jobshop.SchedulingSet as SchedulingSet
import newSimulatorBasedOnJava.rule.operation.basic.FCFS as FCFS
import newSimulatorBasedOnJava.rule.operation.basic.SPT as SPT
import newSimulatorBasedOnJava.rule.operation.basic.WINQ as WINQ
import newSimulatorBasedOnJava.rule.operation.basic.MS as MS
import newSimulatorBasedOnJava.rule.operation.basic.CR as CR
import newSimulatorBasedOnJava.rule.workcenter.basic.WIQ as WIQ
import newSimulatorBasedOnJava.simulation.HeterogeneousSimulation as HeterogeneousSimulation
import newSimulatorBasedOnJava.rule.RuleType as RuleType


class DFJSSMain:
    def __init__(self, **kwargs):
        pass

    def runDynamicSimulation(self, o, utilLevel, sequencingRule, routingRule):
        objectives = []
        objectives.append(o)
        fitnesses = [-1 for i in range(len(objectives))]
        seed = 968356
        set = self.createSchedulingSet(seed, o, utilLevel)
        simulations = set.getSimulations()
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
                    if not ObjValue >= np.Infinity:
                        fitnesses[i] += ObjValue
                for k in range(1, set.getReplications()[j]):
                    # print("run: " + str(k))
                    simulation.rerun()
                    ObjValue = simulation.objectiveValue(objectives[i])
                    if not ObjValue >= np.Infinity:
                        fitnesses[i] += ObjValue
                simulation.reset()

        for i in range(len(fitnesses)):
            fitnesses[i]/=set.getReplications()[0]
            print("objective: " + str(fitnesses[i]))
            print()
        finishTime = time.time()
        duration = finishTime - startTime
        print("duration: " + str(duration))

    def runDynamicSimulationMultiProcess(self, cube):
        o, utilLevel, sequencingRule, routingRule = cube[0], cube[1], cube[2], cube[3]
        objectives = []
        objectives.append(o)
        fitnesses = [-1 for i in range(len(objectives))]
        seed = 968356
        set = self.createSchedulingSet(seed, o, utilLevel)
        simulations = set.getSimulations()
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
                    if not ObjValue >= np.Infinity:
                        fitnesses[i] += ObjValue
                rep = set.getReplications()[j]
                if rep > 1:
                    cores = cpu_count()
                    print("cores: " + str(cores))
                    times = int(rep/cores)
                    remain = int(rep%cores)
                    for c in range(times):
                        cube_parts = [objectives[i] for k in range(1, cores)]
                        # print("objective: " + objectives[i])
                        with Pool(cores) as p:
                            ObjValue = p.map(simulation.rerunMultiProcess, cube_parts)
                            print("objective: " + str(ObjValue))
                            # if not ObjValue >= np.Infinity:
                            #     fitnesses[i] += ObjValue
                    cube_parts = [objectives[i] for k in range(1, remain)]
                    # print("objective: " + objectives[i])
                    with Pool(cores) as p:
                        ObjValue = p.map(simulation.rerunMultiProcess, cube_parts)
                        print("objective: " + str(ObjValue))
                    simulation.reset()

        for i in range(len(fitnesses)):
            fitnesses[i]/=set.getReplications()[0]
            print("objective: " + str(fitnesses[i]))
            print()
        finishTime = time.time()
        duration = finishTime - startTime
        print("duration: " + str(duration))


    def createSchedulingSet(self, seed, objective, utilLevel):
        trainSimulations = []
        replications = []
        objectives = []
        objectives.append(objective)

        numMachines = 10
        numJobs = 5000
        warmupJobs = 1000
        minNumOperations = 1
        maxNumOperations = numMachines
        dueDateFactor = 1.5
        rep = 30

        simulation = HeterogeneousSimulation.HeterogeneousSimulation(seed, None, None, numMachines, numJobs, warmupJobs,
                                             minNumOperations, maxNumOperations,utilLevel, dueDateFactor, False)
        trainSimulations.append(simulation)
        replications.append(rep)
        return SchedulingSet.SchedulingSet(trainSimulations, replications, objectives)


    def multiprocess(self, cube):
        cores = cpu_count()
        print("cores: " + str(cores))

        cube_parts = np.array_split(cube, cores, axis=2)

        with Pool(cores) as p:
            parts = p.map(self.runDynamicSimulationMultiProcess, cube_parts)
            try:
                return np.concatenate(parts, axis=2)
            except ValueError:
                print("Array provided is smaller than # of cores available")






# def main():
if __name__ == '__main__':
    simulationType = "dynamic"
    # sequencingRule = FCFS.FCFS(RuleType.RuleType.SEQUENCING)
    # sequencingRule = WINQ.WINQ(RuleType.RuleType.SEQUENCING)
    # sequencingRule = MS.MS(RuleType.RuleType.SEQUENCING)
    sequencingRule = CR.CR(RuleType.RuleType.SEQUENCING)
    # routingRule = WIQ.WIQ(RuleType.RuleType.ROUTING)
    # sequencingRule = SPT.SPT(RuleType.RuleType.SEQUENCING)
    routingRule = WIQ.WIQ(RuleType.RuleType.ROUTING)

    # objectives = [Objective.Objective.MEAN_TARDINESS, Objective.Objective.MAX_TARDINESS,
    #               Objective.Objective.MEAN_WEIGHTED_TARDINESS]
    objectives = [Objective.Objective.TOTAL_TARDINESS]
    utilLevels = [0.95]
    dfjss = DFJSSMain()
    for i in range(len(utilLevels)):
        utilLevel = utilLevels[i]
        for j in range(len(objectives)):
            o = objectives[j]
            print("Objective: " + o + ", Utilisation level: " + str(utilLevel))
            print("Sequencing rule: " + sequencingRule.getName())
            print("Routing rule: " + routingRule.getName())
            cuba = (o, utilLevel, sequencingRule, routingRule)
            # dfjss.multiprocess(cuba)
            dfjss.runDynamicSimulationMultiProcess(cuba)
            # dfjss.runDynamicSimulation(o, utilLevel, sequencingRule, routingRule)