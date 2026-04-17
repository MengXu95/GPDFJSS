
from newSimulatorBasedOnJava.jobshop.Job import Job
from newSimulatorBasedOnJava.jobshop.Operation import Operation
from newSimulatorBasedOnJava.jobshop.OperationOption import OperationOption
from newSimulatorBasedOnJava.simulation.DynamicSimulation import DynamicSimulation
from newSimulatorBasedOnJava.simulation.event.JobArrivalEvent import JobArrivalEvent
from newSimulatorBasedOnJava.util.random.UniformSampler import UniformSampler
from newSimulatorBasedOnJava.util.random.ExponentialSampler import ExponentialSampler



class HeterogeneousSimulation(DynamicSimulation):
    def __init__(self, seed, sequencingRule, routingRule, numWorkCenters, numJobsRecorded, warmupJobs, minNumOperations, maxNumOperations,
                utilLevel, dueDateFactor, revisit, **kwargs):
        DynamicSimulation.__init__(self, seed, sequencingRule, routingRule, numWorkCenters, numJobsRecorded, warmupJobs, minNumOperations, maxNumOperations,
                utilLevel, dueDateFactor, revisit)

        self.workloadSampler = UniformSampler(100, 1000)
        self.workCenterDistanceSampler = UniformSampler(35, 500)
        self.entryWorkCenterTranserTime = [-1 for i in range(numWorkCenters)]
        self.workCenterExitTranserTime = [-1 for i in range(numWorkCenters)]
        self.workCenterTranserTimeMatrix = [[-1 for i in range(numWorkCenters)] for i in range(numWorkCenters)]

        self.transferRate, self.realMeanTransportationTime, numTransportation = 5, 0, 0

        # for speed up
        i = 0
        while i < numWorkCenters:
            entryTransferDistance = self.workCenterDistanceSampler.next(self.randomDataGenerator)
            exitTransferDistance = self.workCenterDistanceSampler.next(self.randomDataGenerator)
            self.entryWorkCenterTranserTime[i] = entryTransferDistance / self.transferRate
            self.workCenterExitTranserTime[i] = exitTransferDistance / self.transferRate
            self.workCenterTranserTimeMatrix[i][i] = 0
            j = i + 1
            while j < numWorkCenters:
                transferDistance = self.workCenterDistanceSampler.next(self.randomDataGenerator)
                self.workCenterTranserTimeMatrix[i][j] = transferDistance / self.transferRate
                self.workCenterTranserTimeMatrix[j][i] = transferDistance / self.transferRate
                self.realMeanTransportationTime += self.workCenterTranserTimeMatrix[i][j]
                numTransportation = numTransportation + 1
                j += 1
            i += 1

        # original
        # for i in range(numWorkCenters):
        #     entryTransferDistance = self.workCenterDistanceSampler.next(self.randomDataGenerator)
        #     exitTransferDistance = self.workCenterDistanceSampler.next(self.randomDataGenerator)
        #     self.entryWorkCenterTranserTime[i] = entryTransferDistance / self.transferRate
        #     self.workCenterExitTranserTime[i] = exitTransferDistance / self.transferRate
        #     for j in range(i, numWorkCenters):
        #         if i == j:
        #             transferDistance = 0
        #             self.workCenterTranserTimeMatrix[i][j] = transferDistance / self.transferRate
        #         else:
        #             transferDistance = self.workCenterDistanceSampler.next(self.randomDataGenerator)
        #             self.workCenterTranserTimeMatrix[i][j] = transferDistance / self.transferRate
        #             self.workCenterTranserTimeMatrix[j][i] = transferDistance / self.transferRate
        #             self.realMeanTransportationTime += self.workCenterTranserTimeMatrix[i][j]
        #             numTransportation = numTransportation + 1

        self.realMeanTransportationTime = self.realMeanTransportationTime / numTransportation
        self.realMeanProcessingRate = 0
        self.workCenterProcessingRateSampler = UniformSampler(10, 15)
        # for speed up
        i = 0
        while i < len(self.systemState.getWorkCenters()):
            workCenter = self.systemState.getWorkCenters()[i]
            speed = self.workCenterProcessingRateSampler.next(self.randomDataGenerator)
            workCenter.setProcessingRate(speed)
            self.realMeanProcessingRate += speed
            i += 1

        # original
        # for workCenter in self.systemState.getWorkCenters():
        #     speed = self.workCenterProcessingRateSampler.next(self.randomDataGenerator)
        #     workCenter.setProcessingRate(speed)
        #     self.realMeanProcessingRate += speed
        self.realMeanProcessingRate = self.realMeanProcessingRate / numWorkCenters

        mean = self.interArrivalTimeMean(numWorkCenters, minNumOperations, maxNumOperations, utilLevel)
        # print("mean: " + str(mean))
        self.interArrivalTimeSampler = ExponentialSampler(mean)

        if 'sequencingTrain' in kwargs and not kwargs['sequencingTrain']:
            self.setup()

        if 'routingTrain' in kwargs and not kwargs['routingTrain']:
            self.setup()

        if 'sequencingTest' in kwargs and not kwargs['sequencingTest']:
            self.setup()

        if 'routingTest' in kwargs and not kwargs['routingTest']:
            self.setup()

        if 'GPTrain' in kwargs and kwargs['GPTrain']:
            self.setup()


    def interArrivalTimeMean(self, numWorkCenters, minNumOps, maxNumOps, utilLevel):
        meanNumOps = 0.5 * (minNumOps + maxNumOps)
        meanProcTime = self.workloadSampler.getMean()/self.workCenterProcessingRateSampler.getMean()
        meanTransportationTime = self.workCenterDistanceSampler.getLower()/self.transferRate
        return (meanNumOps * (meanProcTime + meanTransportationTime))/ (utilLevel * numWorkCenters)

    def generateJob(self):
        arrivalTime = self.getClockTime() + self.interArrivalTimeSampler.next(self.randomDataGenerator)
        weight = self.jobWeightSampler.next(self.randomDataGenerator)
        job = Job(self.numJobsArrived, [], arrivalTime, arrivalTime, 0, weight)
        numOperations = self.numOperationsSampler.next(self.randomDataGenerator)

        # for speed up
        i = 0
        while i < numOperations:
            o = Operation(job,i)
            numOptions = self.numOperationsSampler.next(self.randomDataGenerator)
            route = self.randomDataGenerator.permutation(self.numWorkCenters, numOptions)
            workload = self.workloadSampler.next(self.randomDataGenerator)
            j = 0
            while j < numOptions:
                workCenter = self.systemState.getWorkCenter(route[j])
                procTimeForThis = workload / workCenter.getProcessingRate()
                o.addOperationOption(OperationOption(o, j, procTimeForThis, workCenter))
                j += 1
            job.addOperation(o)
            i += 1

        # original
        # for i in range(numOperations):
        #     o = Operation.Operation(job,i)
        #     numOptions = self.numOperationsSampler.next(self.randomDataGenerator)
        #     route = self.randomDataGenerator.permutation(self.numWorkCenters, numOptions)
        #     workload = self.workloadSampler.next(self.randomDataGenerator)
        #     for j in range(numOptions):
        #         workCenter = self.systemState.getWorkCenter(route[j])
        #         procTimeForThis = workload / workCenter.getProcessingRate()
        #         o.addOperationOption(OperationOption.OperationOption(o, j, procTimeForThis, workCenter))
        #     job.addOperation(o)

        job.linkOperations()
        totalProcTime = numOperations * self.workloadSampler.getMean() / self.workCenterProcessingRateSampler.getMean()
        dueDate = job.getReleaseTime() + self.dueDateFactor * totalProcTime
        job.setDueDate(dueDate)
        self.systemState.addJobToSystem(job)
        self.numJobsArrived += 1
        # item = JobArrivalEvent.JobArrivalEvent(job)
        item = JobArrivalEvent(job)
        # self.eventQueue.put(item)
        self.eventQueue.push(item, item.time)

    def getEntryWorkCenterTranserTime(self, workCenterID):
        return self.entryWorkCenterTranserTime[workCenterID]

    def getWorkCenterExitTranserTime(self, workCenterID):
        return self.workCenterExitTranserTime[workCenterID]

    def getWorkCenterTranserTime(self, fromWorkCenterID, toWorkCenterID):
        return self.workCenterTranserTimeMatrix[fromWorkCenterID][toWorkCenterID]

    def rotateSeed(self):
        self.seed += self.SEED_ROTATION
        self.reset()

    def standardMissing(self, seed, sequencingRule, routingRule, numWorkCenters, numJobsRecorded, warmupJobs, utilLevel, dueDateFactor):
        return HeterogeneousSimulation(seed, sequencingRule, routingRule, numWorkCenters, numJobsRecorded,
                                warmupJobs, 1, numWorkCenters, utilLevel, dueDateFactor, False)
