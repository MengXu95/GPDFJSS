import numpy
from newSimulatorBasedOnJava.jobshop import Objective
from newSimulatorBasedOnJava.simulation.event import ProcessStartEvent
from newSimulatorBasedOnJava.simulation.state import SystemState

from newSimulatorBasedOnJava.simulation.event import EventPriorityQueue


class Simulation:
    def __init__(self, sequencingRule, routingRule, numWorkCenters, numJobsRecorded, warmupJobs, **kwargs):
        self.sequencingRule, self.routingRule, self.numWorkCenters, self.numJobsRecorded, self.warmupJobs = sequencingRule, routingRule, numWorkCenters, numJobsRecorded, warmupJobs

        self.systemState = SystemState.SystemState(0, [], [], [], None)  # todo: need to check if this is right
        self.systemState.setSimulation(self)
        self.eventQueue = EventPriorityQueue.EventPriorityQueue()

        self.numJobsArrived, self.sequencingRule, self.routingRule, self.count, self.throughput, self.beforeThroughput, self.afterThroughput = 0, None, None, 0, 0, 0, 0

        # self.sequencingRule = sequencingRule
        # self.routingRule = routingRule
        # self.numWorkCenters = numWorkCenters
        # self.numJobsRecorded = numJobsRecorded
        # self.warmupJobs = warmupJobs
        #
        # self.systemState = SystemState.SystemState(0, [], [], [], None)  # todo: need to check if this is right
        # self.systemState.setSimulation(self)
        # self.eventQueue = EventPriorityQueue.EventPriorityQueue()
        # self.numJobsArrived = 0
        # self.sequencingRule = None
        # self.routingRule = None
        # self.count = -1
        # self.throughput = -1
        # self.beforeThroughput = -1
        # self.afterThroughput = -1

    def getNumJobsArrived(self):
        return self.numJobsArrived

    def getNumJobsRecorded(self):
        return self.numJobsRecorded

    def getSequencingRule(self):
        return self.sequencingRule

    def getRoutingRule(self):
        return self.routingRule

    def getWarmupJobs(self):
        return self.warmupJobs

    def getSystemState(self):
        return self.systemState

    def setSequencingRule(self, sequencingRule):
        self.sequencingRule = sequencingRule

    def setRoutingRule(self, routingRule):
        self.routingRule = routingRule
        # need to reset state as well, as the operationoptions associated
        # with workcenters are chosen using this routing rule, so current
        # values are outdated # original by fangfang
        # self.resetState() # todo: hide by mengxu 2022.11.22 for RL training need to double check this

    def getClockTime(self):
        return self.systemState.getClockTime()

    def addEvent(self, event):
        self.eventQueue.push(event,
                             event.getTime())  # todo: need to check this, the priority queue is quite important in this simulation

    def canAddToQueue(self, process):
        if not self.eventQueue.empty():
            a = self.eventQueue._queue[0]  # todo: need to check this, the priority queue is quite important in this simulation
            # self.eventQueue.push(a, a.getTime()) # add by meng
            if isinstance(a, ProcessStartEvent.ProcessStartEvent):
                if a.getProcess().getWorkCenter().getId() == process.getWorkCenter().getId():
                    return 0  # for speed up by mengxu
                    # return False # original
        return 1
        # return True # original

    def run(self):
        self.count = 0
        while not self.eventQueue.empty() and self.throughput < self.warmupJobs + self.numJobsRecorded:
            nextEvent = self.eventQueue.pop()
            self.beforeThroughput = self.throughput
            self.systemState.setClockTime(nextEvent.getTime())
            nextEvent.trigger(self)
            self.afterThroughput = self.throughput
            if self.throughput > self.warmupJobs and self.afterThroughput - self.beforeThroughput == 0:
                self.count = self.count + 1
            if self.count > 100000:
                self.count = 0
                self.systemState.setClockTime(numpy.inf)
                # print("bad run: self.count > 100000")
                self.eventQueue.clear()
            i = 0
            while i < len(self.systemState.getWorkCenters()):
                workCenter = self.systemState.workCenters[i]
                numOpsInQueue = workCenter.numOpsInQueue()
                if numOpsInQueue > 100:
                    self.systemState.setClockTime(numpy.inf)
                    self.eventQueue.clear()
                i += 1
            # original
            # for workCenter in self.systemState.getWorkCenters():
            #     numOpsInQueue = workCenter.numOpsInQueue()
            #     if numOpsInQueue > 100:
            #         self.systemState.setClockTime(numpy.inf)
            #         self.eventQueue.clear()
            #         # print("bad run: " + str(workCenter.getId()) + ": workCenter.numOpsInQueue() > 100")

    def runWarmUpForDataCollection(self, sequencing_rule, routing_rule):
        self.count = 0
        #todo: need to use default rule like: PT and WIQ
        self.setSequencingRule(sequencing_rule)
        self.setRoutingRule(routing_rule)
        while not self.eventQueue.empty() and self.throughput < self.warmupJobs:
            nextEvent = self.eventQueue.pop()
            self.beforeThroughput = self.throughput
            self.systemState.setClockTime(nextEvent.getTime())
            nextEvent.trigger(self)
            self.afterThroughput = self.throughput
            if self.throughput > self.warmupJobs and self.afterThroughput - self.beforeThroughput == 0:
                self.count = self.count + 1
            if self.count > 100000:
                self.count = 0
                self.systemState.setClockTime(numpy.inf)
                # print("bad run: self.count > 100000")
                self.eventQueue.clear()
            i = 0
            while i < len(self.systemState.getWorkCenters()):
                workCenter = self.systemState.workCenters[i]
                numOpsInQueue = workCenter.numOpsInQueue()
                if numOpsInQueue > 100:
                    self.systemState.setClockTime(numpy.inf)
                    self.eventQueue.clear()
                i += 1
        # todo: need to set back to the GP rules for every individual
        # self.setSequencingRule("PT")
        # self.setRoutingRule("WIQ")
        # print(f"Done warmup! with {self.throughput} of {self.warmupJobs}")


    def rerun(self):
        self.resetState()
        self.run()

    def rerunMultiProcess(self, objective):
        self.resetState()
        self.run()
        return self.objectiveValue(objective)

    def completeJob(self, job):
        if self.warmupJobs <= job.getId() < self.numJobsRecorded + self.warmupJobs:
            self.throughput = self.throughput + 1
            self.count = 0
            self.systemState.addCompletedJob(job)
        elif job.getId() < self.warmupJobs:
            self.throughput = self.throughput + 1
            self.count = 0
            self.systemState.warmupJobsCompleted.append(job)  # add by mengxu for RL 2022.11.21
        self.systemState.removeJobFromSystem(job)


    def makespan(self):
        if len(self.systemState.getJobsCompleted()) < self.numJobsRecorded:
            return numpy.inf
        value = 0
        i = 0
        while i < len(self.systemState.getJobsCompleted()):
            job = self.systemState.jobsCompleted[i]
            if job is None:
                return numpy.inf
            else:
                tmp = job.getCompletionTime()
                value += tmp
            i += 1
        return value

        # original
        # for job in self.systemState.getJobsCompleted():
        #     if job is None:
        #         return numpy.inf
        #     else:
        #         tmp = job.getCompletionTime()
        #         value += tmp
        # return value

    def meanFlowtime(self):
        if len(self.systemState.getJobsCompleted()) < self.numJobsRecorded:
            return numpy.inf
        value = 0
        i = 0
        while i < len(self.systemState.getJobsCompleted()):
            job = self.systemState.jobsCompleted[i]
            if job is None:
                return numpy.inf
            else:
                tmp = job.flowTime()
                value += tmp
            i += 1
        return value / len(self.systemState.getJobsCompleted())

        # original
        # for job in self.systemState.getJobsCompleted():
        #     if job is None:
        #         return numpy.inf
        #     else:
        #         tmp = job.flowTime()
        #         value += tmp
        # return value / len(self.systemState.getJobsCompleted())

    def maxFlowtime(self):
        if len(self.systemState.getJobsCompleted()) < self.numJobsRecorded:
            return numpy.inf
        value = 0
        i = 0
        while i < len(self.systemState.getJobsCompleted()):
            job = self.systemState.jobsCompleted[i]
            if job is None:
                return numpy.inf
            else:
                tmp = job.flowTime()
                if value < tmp:
                    value = tmp
            i += 1
        return value

        # original
        # for job in self.systemState.getJobsCompleted():
        #     if job is None:
        #         return numpy.inf
        #     else:
        #         tmp = job.flowTime()
        #         if value < tmp:
        #             value = tmp
        # return value

    def meanWeightedFlowtime(self):
        if len(self.systemState.getJobsCompleted()) < self.numJobsRecorded:
            return numpy.inf
        value = 0
        i = 0
        while i < len(self.systemState.getJobsCompleted()):
            job = self.systemState.jobsCompleted[i]
            if job is None:
                return numpy.inf
            else:
                tmp = job.weightedFlowTime()
                value += tmp
            i += 1
        return value / len(self.systemState.getJobsCompleted())

        # original
        # for job in self.systemState.getJobsCompleted():
        #     if job is None:
        #         return numpy.inf
        #     else:
        #         tmp = job.weightedFlowTime()
        #         value += tmp
        # return value / len(self.systemState.getJobsCompleted())

    def maxWeightedFlowtime(self):
        if len(self.systemState.getJobsCompleted()) < self.numJobsRecorded:
            return numpy.inf
        value = 0
        i = 0
        while i < len(self.systemState.getJobsCompleted()):
            job = self.systemState.jobsCompleted[i]
            if job is None:
                return numpy.inf
            else:
                tmp = job.weightedFlowTime()
                if value < tmp:
                    value = tmp
            i += 1
        return value

        # original
        # for job in self.systemState.getJobsCompleted():
        #     if job is None:
        #         return numpy.inf
        #     else:
        #         tmp = job.weightedFlowTime()
        #         if value < tmp:
        #             value = tmp
        # return value

    def meanTardiness(self):
        if len(self.systemState.getJobsCompleted()) < self.numJobsRecorded:
            return numpy.inf
        value = 0
        i = 0
        while i < len(self.systemState.getJobsCompleted()):
            job = self.systemState.jobsCompleted[i]
            if job is None:
                return numpy.inf
            else:
                tmp = job.tardiness()
                value += tmp
            i += 1
        return value / numpy.float64(len(self.systemState.getJobsCompleted()))

        # original
        # for job in self.systemState.getJobsCompleted():
        #     if job is None:
        #         return numpy.inf
        #     else:
        #         tmp = job.tardiness()
        #         value += tmp
        # return value / len(self.systemState.getJobsCompleted())

    def totalTardiness(self):
        if len(self.systemState.getJobsCompleted()) < self.numJobsRecorded:
            return numpy.inf
        value = 0
        i = 0
        while i < len(self.systemState.getJobsCompleted()):
            job = self.systemState.jobsCompleted[i]
            if job is None:
                return numpy.inf
            else:
                tmp = job.tardiness()
                value += tmp
            i += 1
        return value

        # original
        # for job in self.systemState.getJobsCompleted():
        #     if job is None:
        #         return numpy.inf
        #     else:
        #         tmp = job.tardiness()
        #         value += tmp
        # return value

    def maxTardiness(self):
        if len(self.systemState.getJobsCompleted()) < self.numJobsRecorded:
            return numpy.inf
        value = 0
        i = 0
        while i < len(self.systemState.getJobsCompleted()):
            job = self.systemState.jobsCompleted[i]
            if job is None:
                return numpy.inf
            else:
                tmp = job.tardiness()
                if value < tmp:
                    value = tmp
            i += 1
        return value

        # original
        # for job in self.systemState.getJobsCompleted():
        #     if job is None:
        #         return numpy.inf
        #     else:
        #         tmp = job.tardiness()
        #         if value < tmp:
        #             value = tmp
        # return value

    def meanWeightedTardiness(self):
        if len(self.systemState.getJobsCompleted()) < self.numJobsRecorded:
            return numpy.inf
        value = 0
        i = 0
        while i < len(self.systemState.getJobsCompleted()):
            job = self.systemState.jobsCompleted[i]
            if job is None:
                return numpy.inf
            else:
                tmp = job.weightedTardiness()
                value += tmp
            i += 1
        return value / len(self.systemState.getJobsCompleted())

        # original
        # for job in self.systemState.getJobsCompleted():
        #     if job is None:
        #         return numpy.inf
        #     else:
        #         tmp = job.weightedTardiness()
        #         value += tmp
        # return value / len(self.systemState.getJobsCompleted())

    def maxWeightedTardiness(self):
        if len(self.systemState.getJobsCompleted()) < self.numJobsRecorded:
            return numpy.inf
        value = 0
        i = 0
        while i < len(self.systemState.getJobsCompleted()):
            job = self.systemState.jobsCompleted[i]
            if job is None:
                return numpy.inf
            else:
                tmp = job.weightedTardiness()
                if value < tmp:
                    value = tmp
            i += 1
        return value

        # original
        # for job in self.systemState.getJobsCompleted():
        #     if job is None:
        #         return numpy.inf
        #     else:
        #         tmp = job.weightedTardiness()
        #         if value < tmp:
        #             value = tmp
        # return value

    def objectiveValue(self, objective):
        if objective == Objective.Objective.MAKESPAN:
            return self.makespan()
        if objective == Objective.Objective.MEAN_FLOWTIME:
            return self.meanFlowtime()
        if objective == Objective.Objective.MAX_FLOWTIME:
            return self.maxFlowtime()
        if objective == Objective.Objective.MEAN_WEIGHTED_FLOWTIME:
            return self.meanWeightedFlowtime()
        if objective == Objective.Objective.MAX_WEIGHTED_FLOWTIME:
            return self.maxWeightedFlowtime()
        if objective == Objective.Objective.MEAN_TARDINESS:
            return self.meanTardiness()
        if objective == Objective.Objective.TOTAL_TARDINESS:
            return self.totalTardiness()
        if objective == Objective.Objective.MAX_TARDINESS:
            return self.maxTardiness()
        if objective == Objective.Objective.MEAN_WEIGHTED_TARDINESS:
            return self.meanWeightedTardiness()
        if objective == Objective.Objective.MAX_WEIGHTED_TARDINESS:
            return self.maxWeightedTardiness()

        return -1

    def workCenterUtilLevel(self, idx):
        return self.systemState.getWorkCenter(idx).getBusyTime() / self.getClockTime()

    def setup(self):
        pass

    def resetState(self):
        pass

    def reset(self):
        pass

    def rotateSeed(self):
        pass

    def generateJob(self):
        pass
