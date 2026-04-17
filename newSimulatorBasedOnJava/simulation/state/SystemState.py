
class SystemState:
    def __init__(self, clockTime, workCenters, jobsInSystem, jobsCompleted, simulation, **kwargs):
        self.clockTime = clockTime
        self.workCenters = workCenters
        self.jobsInSystem = jobsInSystem
        self.jobsCompleted = jobsCompleted
        self.warmupJobsCompleted = []
        self.simulation = simulation

    def getClockTime(self):
        return self.clockTime

    def getWorkCenters(self):
        return self.workCenters

    def getWorkCenter(self, idx):
        return self.workCenters[idx]

    def getJobsInSystem(self):
        return self.jobsInSystem

    def getNumJobsInSystem(self):
        return len(self.jobsInSystem)

    def getJobsCompleted(self):
        return self.jobsCompleted

    def setClockTime(self, clockTime):
        self.clockTime = clockTime

    def setWorkCenters(self, workCenters):
        self.workCenters = workCenters

    def addWorkCenter(self, workCenter):
        self.workCenters.append(workCenter)

    def addJobToSystem(self, job):
        self.jobsInSystem.append(job)
        # print("job " + str(job.getId()) + " arrived!")

    def removeJobFromSystem(self, job):
        # print("job ID: " + str(job.getId()))
        self.jobsInSystem.remove(job)

    def addCompletedJob(self, job):
        self.jobsCompleted.append(job)

    def setSimulation(self, simulation):
        self.simulation = simulation

    def getSimulation(self):
        return self.simulation

    def getRealTardyRate(self):
        tardyNum = 0
        for job in self.jobsCompleted:
            if job.completionTime > job.dueDate:
                tardyNum += 1
        for job in self.warmupJobsCompleted:
            if job.completionTime > job.dueDate:
                tardyNum += 1
        for job in self.jobsInSystem:
            if self.clockTime > job.dueDate:
                tardyNum += 1
        return tardyNum/self.simulation.numJobsArrived

    def getExpTardyRate(self):
        tardyNum = 0
        for job in self.jobsCompleted:
            if job.completionTime > job.dueDate:
                tardyNum += 1
        for job in self.warmupJobsCompleted:
            if job.completionTime > job.dueDate:
                tardyNum += 1
        for job in self.jobsInSystem:
            if self.clockTime > job.dueDate or job.getSlack(self.clockTime) < 0:
                tardyNum += 1
        return tardyNum/self.simulation.numJobsArrived

    def getTotalBusyTime(self):
        totalBusyTime = 0
        for workCenter in self.workCenters:
            totalBusyTime = totalBusyTime + workCenter.getBusyTime()
        return totalBusyTime

    def getWorkInSystem(self):
        totalProcessTimeInSystem = 0
        for workCenter in self.workCenters:
            totalProcessTimeInSystem = totalProcessTimeInSystem + workCenter.getWorkInQueue()
        return totalProcessTimeInSystem

    def getNumOfOperationInSystem(self):
        totalNumOfOperationInSystem = 0
        for workCenter in self.workCenters:
            totalNumOfOperationInSystem = totalNumOfOperationInSystem + workCenter.numOpsInQueue()
        return totalNumOfOperationInSystem

    def getNumOfOperationInWholeSystem(self):
        totalNumOfOperationInSystem = 0
        for job in self.jobsInSystem:
            totalNumOfOperationInSystem = totalNumOfOperationInSystem + len(job.getOperations()) - len(job.getProcessFinishEvents())
        return totalNumOfOperationInSystem

    def getMinWorkInNextQueue(self, op):
        next = op.getNext()
        if next == None:
            return 0
        return next.getLeastWorkLoad()

    def getMaxWorkInNextQueue(self, op):
        next = op.getNext()
        if next == None:
            return 0
        return next.getMaxWorkLoad()

    def getAvgWorkInNextQueue(self, op):
        next = op.getNext()
        if next == None:
            return 0
        return next.getAvgWorkLoad()

    def getMinNumOperationInNextQueue(self, op):
        next = op.getNext()
        if next == None:
            return 0
        return next.getLeastNumOfOperation()

    def getMaxNumOperationInNextQueue(self, op):
        next = op.getNext()
        if next == None:
            return 0
        return next.getMaxNumOfOperation()

    def getAveNumOperationInNextQueue(self, op):
        next = op.getNext()
        if next == None:
            return 0
        return next.getAveNumOfOperation()

    def getMinNextProcessTime(self, op):
        next = op.getNext()
        if next == None:
            return 0
        return next.getLeastProcessTime()

    def getMaxNextProcessTime(self, op):
        next = op.getNext()
        if next == None:
            return 0
        return next.getMaxProcessTime()

    def getMedianNextProcessTime(self, op):
        next = op.getNext()
        if next == None:
            return 0
        return next.getMedianProcessTime()

    def reset(self):
        self.clockTime = 0.0
        for workCenter in self.workCenters:
            workCenter.reset()
        self.jobsInSystem.clear()
        self.jobsCompleted.clear()
        self.warmupJobsCompleted.clear()

        # if self.simulation.useLS:
        #     self.jobsCompleted = [None for i in self.simulation.getNumJobsRecorded] #todo: need to check if this is right
        # else:
        #     self.jobsCompleted.clear()

    def slack(self, operation):
        return operation.getOperation().getJob().getDueDate()- self.getClockTime() - operation.getWorkRemaining()


    def clone(self):
        clonedWCs = []
        for wc in self.workCenters:
            clonedWCs.append(wc.clone())
        return SystemState(self.clockTime, clonedWCs, [], [], self.simulation)