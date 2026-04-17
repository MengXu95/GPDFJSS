from numpy import sort

'''
A job
created by Mengxu on 2022.11.14
'''


class Job:
    def __init__(self, id, operation_list, arrivalTime, releaseTime, dueDate, weight, **kwargs):
        self.id = id
        self.operation_list = operation_list
        self.arrivalTime = arrivalTime
        self.releaseTime = releaseTime
        self.dueDate = dueDate
        self.weight = weight
        self.processFinishEvents = []

        self.totalProcTime = -1
        self.avgProcTime = -1
        self.completionTime = -1

    def getId(self):
        return self.id

    def getOperation_list(self):
        return self.operation_list

    def getProcessFinishEvents(self):
        return self.processFinishEvents

    def getCompletionRate(self): # add by mengxu for RL 2022.11.17
        return len(self.processFinishEvents)/len(self.operation_list)

    def addProcessFinishEvent(self, processFinishEvent):
        self.processFinishEvents.append(processFinishEvent)

    def getOperation(self, idx):
        return self.operation_list[idx]

    def getArrivalTime(self):
        return self.arrivalTime

    def getReleaseTime(self):
        return self.releaseTime

    def getSlack(self, clockTime):
        return self.dueDate - clockTime - self.getRemainingProcessingTime()

    def getDueDate(self):
        return self.dueDate

    def getWeight(self):
        return self.weight

    def getTotalProcTime(self):
        return self.totalProcTime

    def getAvgProcTime(self):
        return self.avgProcTime

    def getCompletionTime(self):
        return self.completionTime

    def setDueDate(self, dueDate):
        self.dueDate = dueDate

    def setCompletionTime(self, completionTime):
        self.completionTime = completionTime

    def flowTime(self):
        return self.completionTime - self.arrivalTime

    def weightedFlowTime(self):
        return self.weight * self.flowTime()

    def tardiness(self):
        tardiness = self.completionTime - self.dueDate
        if tardiness < 0:
            return 0
        return tardiness

    def weightedTardiness(self):
        return self.weight * self.tardiness()

    def addOperation(self, operation):
        self.operation_list.append(operation)

    # def calculateEstimatedSlackWhenArrival(self):
    #     estimated_slack_time = self.due_list[0] - self.env.now - self.least_waiting - np.sum(remaining_ptl.max(axis=1))

    def getRemainingProcessingTime(self):
        remainingProcessingTime = 0
        numFinishedOperation = len(self.processFinishEvents)
        for i in range(numFinishedOperation, len(self.operation_list)):
            operation = self.operation_list[i]
            remainingProcessingTime += operation.getMedianProcessTime()
        return remainingProcessingTime


    def linkOperations(self):
        # I think should also consider the transfer time by mengxu 2022.11.18
        next = None
        nextProcTime = 0
        workRemaining = 0
        numOpsRemaining = 0
        for i in range(len(self.operation_list) - 1, -1, -1):
            operation = self.operation_list[i]
            medianProcTime = -1
            # for improve speed of simulator by mengxu 2022.11.24
            procTimes = [j.getProcTime() for j in operation.getOperationOptions()]

            # original
            # procTimes = []
            # for j in range(0, 1, len(operation.getOperationOptions())):
            #     procTimes.append(operation.getOperationOptions()[j].getProcTime())

            procTimes = sort(procTimes)
            if len(procTimes) % 2 == 0:
                idx = int(len(procTimes) / 2)
                medianProcTime = (procTimes[idx] + procTimes[idx - 1]) / 2
            else:
                idx = int(len(procTimes) / 2) # todo: need to double check this part
                medianProcTime = procTimes[idx]

            for option in operation.getOperationOptions():
                option.setWorkRemaining(workRemaining + medianProcTime)
                option.setNumOpsRemaining(numOpsRemaining)
                option.setNextProcTime(nextProcTime)

            numOpsRemaining = numOpsRemaining + 1
            workRemaining += medianProcTime

            operation.setNext(next)

            next = operation
            nextProcTime = medianProcTime

        self.totalProcTime = workRemaining
        self.avgProcTime = self.totalProcTime / len(self.operation_list)

    def equals(self, other):
        return self.id == other.id

    def __cmp__(self, other):
        if self.arrivalTime < other.arrivalTime:
            return -1

        if self.arrivalTime > other.arrivalTime:
            return 1

        return 0
