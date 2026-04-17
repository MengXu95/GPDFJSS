import time

import numpy as np

import newSimulatorBasedOnJava.jobshop.Objective as Objective
import newSimulatorBasedOnJava.jobshop.SchedulingSet as SchedulingSet
import newSimulatorBasedOnJava.rule.operation.basic.FCFS as FCFS
import newSimulatorBasedOnJava.rule.operation.basic.SPT as SPT
import newSimulatorBasedOnJava.rule.workcenter.basic.WIQ as WIQ
import newSimulatorBasedOnJava.simulation.HeterogeneousSimulation as HeterogeneousSimulation
import newSimulatorBasedOnJava.rule.RuleType as RuleType
import newSimulatorBasedOnJava.RL.routingRuleTrain.SimulationForRoutingRuleTrain as SimulationForRoutingRuleTrain


class DFJSSMain_RL_R_train:
    def __init__(self, **kwargs):
        pass

    def runDynamicSimulation(self, o, utilLevel, sequencingRule, routingRule, run_seed):
        objectives = []
        objectives.append(o)
        fitnesses = [-1 for i in range(len(objectives))]
        seed = 968356
        set = self.createSchedulingSet(seed, o, utilLevel, run_seed)
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




    def createSchedulingSet(self, seed, objective, utilLevel, run_seed):
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
        rep = 1

        simulation = SimulationForRoutingRuleTrain.SimulationForRoutingRuleTrain(seed, None, None, numMachines, numJobs, warmupJobs,
                                                                     minNumOperations, maxNumOperations, utilLevel,
                                                                     dueDateFactor, False, objective = objective, run_seed = run_seed, routingTest=False)

        # simulation = HeterogeneousSimulation.HeterogeneousSimulation(seed, None, None, numMachines, numJobs, warmupJobs,
        #                                      minNumOperations, maxNumOperations,utilLevel, dueDateFactor, False)
        trainSimulations.append(simulation)
        replications.append(rep)
        return SchedulingSet.SchedulingSet(trainSimulations, replications, objectives)






# def main():
if __name__ == '__main__':
    simulationType = "dynamic"
    sequencingRule = FCFS.FCFS(RuleType.RuleType.SEQUENCING)
    # routingRule = WIQ.WIQ(RuleType.RuleType.ROUTING)
    # sequencingRule = SPT.SPT(RuleType.RuleType.SEQUENCING)
    routingRule = WIQ.WIQ(RuleType.RuleType.ROUTING)

    # objectives = [Objective.Objective.MEAN_TARDINESS, Objective.Objective.MAX_TARDINESS,
    #               Objective.Objective.MEAN_WEIGHTED_TARDINESS]
    objectives = [Objective.Objective.TOTAL_TARDINESS]
    utilLevels = [0.95]
    run_seed = 0
    dfjss = DFJSSMain_RL_R_train()
    for i in range(len(utilLevels)):
        utilLevel = utilLevels[i]
        for j in range(len(objectives)):
            o = objectives[j]
            print("Objective: " + o + ", Utilisation level: " + str(utilLevel))
            print("Sequencing rule: " + sequencingRule.getName())
            print("Routing rule: RL action")
            dfjss.runDynamicSimulation(o, utilLevel, sequencingRule, routingRule, run_seed)