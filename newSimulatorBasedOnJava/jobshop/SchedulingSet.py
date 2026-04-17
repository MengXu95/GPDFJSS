from newSimulatorBasedOnJava.simulation import HeterogeneousSimulation


class SchedulingSet:
    def __init__(self, simulations, replications, objectives, **kwargs):
        self.simulations = simulations
        self.replications = replications


    def getSimulations(self):
        return self.simulations

    def getReplications(self):
        return self.replications

    def setReplications(self, replications):
        self.replications = replications

    def rotateSeed(self):
        for simulation in self.simulations:
            simulation.rotateSeed()

    def dynamicMissingSet(self, simSeed, utilLevel, dueDateFactor, objectives, reps):
        simulations = []
        simulations.append(HeterogeneousSimulation.HeterogeneousSimulation.standardMissing(simSeed, None, None, 10, 5000, 1000,
                        utilLevel, dueDateFactor))
        replications = []
        replications.append(reps)
        return SchedulingSet(simulations, replications, objectives)

    

