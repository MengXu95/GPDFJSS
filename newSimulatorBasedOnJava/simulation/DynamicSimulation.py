from numpy import random

from newSimulatorBasedOnJava.jobshop.Job import Job
from newSimulatorBasedOnJava.jobshop.Operation import Operation
from newSimulatorBasedOnJava.jobshop.OperationOption import OperationOption
from newSimulatorBasedOnJava.jobshop.WorkCenter import WorkCenter
from newSimulatorBasedOnJava.simulation.Simulation import Simulation
from newSimulatorBasedOnJava.simulation.event.JobArrivalEvent import JobArrivalEvent
from newSimulatorBasedOnJava.util.random.TwoSixTwoSampler import TwoSixTwoSampler
from newSimulatorBasedOnJava.util.random.UniformIntegerSampler import UniformIntegerSampler
from newSimulatorBasedOnJava.simulation import HeterogeneousSimulation
from newSimulatorBasedOnJava.util.random.ExponentialSampler import ExponentialSampler


class DynamicSimulation(Simulation):
    def __init__(self, seed, sequencingRule, routingRule, numWorkCenters, numJobsRecorded, warmupJobs, minNumOperations,
                 maxNumOperations, utilLevel, dueDateFactor, revisit, **kwargs):
        Simulation.__init__(self, sequencingRule, routingRule, numWorkCenters, numJobsRecorded, warmupJobs)
        self.seed, self.SEED_ROTATION = seed, 10000
        self.randomDataGenerator = random.default_rng(seed)
        # self.randomDataGenerator.integers()

        self.minNumOperations, self.maxNumOperations, self.utilLevel, self.dueDateFactor, self.revisit = minNumOperations, maxNumOperations, utilLevel, dueDateFactor, revisit

        numOperationsSampler = UniformIntegerSampler(minNumOperations, maxNumOperations)
        self.numOperationsSampler = numOperationsSampler
        self.procTimeSampler = random.uniform(1, 99)
        if not isinstance(self, HeterogeneousSimulation.HeterogeneousSimulation):
            mean = self.interArrivalTimeMean(numWorkCenters, minNumOperations, maxNumOperations, utilLevel)
            self.interArrivalTimeSampler = ExponentialSampler(mean)  # todo: need to check if this is right
        self.jobWeightSampler = TwoSixTwoSampler()  # todo: this should be two six two sampler, need to modify this

        # for speed up
        i = 0
        while i < numWorkCenters:
            self.systemState.addWorkCenter(WorkCenter(i))
            i += 1
        # original
        # for i in range(numWorkCenters):
        #     self.systemState.addWorkCenter(WorkCenter.WorkCenter(i))

        if not isinstance(self, HeterogeneousSimulation.HeterogeneousSimulation):
            self.setup()

    def interArrivalTimeMean(self, numWorkCenters, minNumOps, maxNumOps, utilLevel):
        meanNumOps = 0.5 * (minNumOps + maxNumOps)
        meanProcTime = (1 + 99) / 2
        return (meanNumOps * meanProcTime) / (utilLevel * numWorkCenters)

    def getSeed(self):
        return self.seed

    def getNumWorkCenters(self):
        return self.numWorkCenters

    def getNumJobsRecorded(self):
        return self.numJobsRecorded

    def getWarmupJobs(self):
        return self.warmupJobs

    def getMinNumOperations(self):
        return self.minNumOperations

    def getMaxNumOperations(self):
        return self.maxNumOperations

    def getUtilLevel(self):
        return self.utilLevel

    def getDueDateFactor(self):
        return self.dueDateFactor

    def setup(self):
        self.numJobsArrived = 0
        self.throughput = 0
        self.generateJob()

    def resetState(self):
        self.systemState.reset()
        self.eventQueue.clear()
        # self.x_Axis.clear()
        # self.y_Axis_List.clear()
        # self.y_WorkCenter_Axis_List.clear()
        # self.sequencingDecisions.clear()
        # self.routingDecisions.clear()
        # self.AllDecisions.clear()
        self.setup()

    def reset(self):
        self.reseed(self.seed)
        self.resetState()

    def reseed(self, seed):
        self.seed = seed
        self.randomDataGenerator = random.default_rng(seed)  # todo: need to check if this is right

    def rotateSeed(self):
        self.seed += self.SEED_ROTATION
        self.reset()

    def generateJob(self):
        arrivalTime = self.getClockTime() + self.interArrivalTimeSampler.next(self.randomDataGenerator)
        weight = self.jobWeightSampler.next(self.randomDataGenerator)
        job = Job(self.numJobsArrived, [], arrivalTime, arrivalTime, 0, weight)
        numOperations = self.numOperationsSampler.next(self.randomDataGenerator)

        # for speed up
        i = 0
        while i < numOperations:
            o = Operation(job, i)
            numOptions = self.numOperationsSampler.next(self.randomDataGenerator)
            route = self.randomDataGenerator.permutation(self.numWorkCenters,
                                                         numOptions)  # todo: need to check if this is right
            procTime = self.procTimeSampler.next(self.randomDataGenerator)
            j = 0
            while j < numOptions:
                o.addOperationOption(
                    OperationOption(o, j, procTime, self.systemState.getWorkCenter(route[j])))
                j += 1
            job.addOperation(o)
            i += 1

        # original
        # for i in range(numOperations):
        #     o = Operation.Operation(job, i)
        #     numOptions = self.numOperationsSampler.next(self.randomDataGenerator)
        #     route = self.randomDataGenerator.permutation(self.numWorkCenters, numOptions) # todo: need to check if this is right
        #     procTime = self.procTimeSampler.next(self.randomDataGenerator)
        #     for j in range(numOptions):
        #         o.addOperationOption(OperationOption.OperationOption(o, j, procTime, self.systemState.getWorkCenter(route[j])))
        #     job.addOperation(o)

        job.linkOperations()
        totalProcTime = numOperations * self.procTimeSampler.getMean()
        dueDate = job.getReleaseTime() + self.dueDateFactor * totalProcTime
        job.setDueDate(dueDate)
        self.systemState.addJobToSystem(job)
        self.numJobsArrived += 1
        # self.eventQueue.put(JobArrivalEvent.JobArrivalEvent(job))
        item = JobArrivalEvent(job)
        self.eventQueue.push(item, item.time)

    def sequencingDecisionSituations(self, minQueueLength):
        sequencingDecisionSituations = []
        self.count = 0
        while not self.eventQueue.empty() and self.throughput < self.numJobsRecorded:
            nextEvent = self.eventQueue.get()
            self.beforeThroughput = self.throughput
            self.systemState.setClockTime(nextEvent.getTime())
            nextEvent.addSequencingDecisionSituation(self, sequencingDecisionSituations, minQueueLength)
            self.afterThroughput = self.throughput
            if self.throughput > self.warmupJobs and self.afterThroughput - self.beforeThroughput == 0:
                self.count = self.count + 1
            if self.count > 100000:
                self.count = 0
                self.systemState.setClockTime(numpy.inf)
                self.eventQueue.clear()
                print("bad run reason count > 100000!")

            # for speed up
            i = 0
            while i < len(self.systemState.getWorkCenters()):
                workCenter = self.systemState.getWorkCenters()[i]
                if workCenter.numOpsInQueue() > 100:
                    self.systemState.setClockTime(numpy.inf)
                    self.eventQueue.clear()
                    print("bad run reason w.numOpsInQueue() > 100!")
                i += 1
            # original
            # for workCenter in self.systemState.getWorkCenters():
            #     if workCenter.numOpsInQueue() > 100:
            #         self.systemState.setClockTime(numpy.inf)
            #         self.eventQueue.clear()
            #         print("bad run reason w.numOpsInQueue() > 100!")
        self.resetState()
        return sequencingDecisionSituations

    def routingDecisionSituations(self, minQueueLength):
        routingDecisionSituations = []
        self.count = 0
        while not self.eventQueue.empty() and self.throughput < self.numJobsRecorded:
            nextEvent = self.eventQueue.get()
            self.beforeThroughput = self.throughput
            self.systemState.setClockTime(nextEvent.getTime())
            nextEvent.addRoutingDecisionSituation(self, routingDecisionSituations, minQueueLength)
            self.afterThroughput = self.throughput
            if self.throughput > self.warmupJobs and self.afterThroughput - self.beforeThroughput == 0:
                self.count = self.count + 1
            if self.count > 100000:
                self.count = 0
                self.systemState.setClockTime(numpy.inf)
                self.eventQueue.clear()
                print("bad run reason count > 100000!")
            # for speed up
            i = 0
            while i < len(self.systemState.getWorkCenters()):
                workCenter = self.systemState.getWorkCenters()[i]
                if workCenter.numOpsInQueue() > 100:
                    self.systemState.setClockTime(numpy.inf)
                    self.eventQueue.clear()
                    print("bad run reason w.numOpsInQueue() > 100!")
                i += 1
            # original
            # for workCenter in self.systemState.getWorkCenters():
            #     if workCenter.numOpsInQueue() > 100:
            #         self.systemState.setClockTime(numpy.inf)
            #         self.eventQueue.clear()
            #         print("bad run reason w.numOpsInQueue() > 100!")
        self.resetState()
        return routingDecisionSituations

    def standardMissing(self, seed, sequencingRule, routingRule, numWorkCenters, numJobsRecorded, warmupJobs, utilLevel,
                        dueDateFactor):
        return DynamicSimulation(seed, sequencingRule, routingRule, numWorkCenters, numJobsRecorded,
                                 warmupJobs, 1, numWorkCenters, utilLevel, dueDateFactor, False)
