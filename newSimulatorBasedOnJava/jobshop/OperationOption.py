
class OperationOption:

    def __init__(self, operation, optionId, procTime, workCenter, **kwargs):
        self.operation = operation
        self.optionId = optionId
        self.procTime = procTime
        self.workCenter = workCenter

        self.selectedWorkCenter = None
        self.uploadTime = -1
        self.transferTime = -1
        self.downloadTime = -1
        self.readyTime = -1
        self.workRemaining = -1
        self.numOpsRemaining = -1
        self.flowDueDate = -1
        self.nextProcTime = -1
        self.priority = -1

    def getOperation(self):
        return self.operation

    def getJob(self):
        return self.operation.getJob()

    def getOptionId(self):
        return self.optionId

    def getProcTime(self):
        return self.procTime

    def getWorkCenter(self):
        return self.workCenter

    def getNextOperation(self):
        return self.operation.getNext()

    def getNextOperationOption(self, systemState, routingRule):
        nextOperation = self.operation.getNext()
        if nextOperation != None:
            return nextOperation.chooseOperationOption(systemState, routingRule)
        return None

    def getReadyTime(self):
        return self.readyTime

    def getJob(self):
        return self.operation.getJob()

    def getWorkRemaining(self):
        return self.workRemaining

    def getNumOpsRemaining(self):
        return self.numOpsRemaining

    def getFlowDueDate(self):
        return self.flowDueDate

    def getNextProcTime(self):
        return self.nextProcTime

    def getPriority(self):
        return self.priority

    def setWorkRemaining(self, workRemaining):
        self.workRemaining = workRemaining

    def setNumOpsRemaining(self, numOpsRemaining):
        self.numOpsRemaining = numOpsRemaining

    def setReadyTime(self, readyTime):
        self.readyTime = readyTime

    def setFlowDueDate(self, flowDueDate):
        self.flowDueDate = flowDueDate

    def setNextProcTime(self, nextProcTime):
        self.nextProcTime = nextProcTime

    def getNext(self, systemState, routingRule):
        nextOperation = self.operation.getNext()
        if nextOperation is not None:
            return nextOperation.chooseOperationOption(systemState, routingRule)
        return None


    # def getEarliestExecutionStartTime(self):
    #     if self.operation.getId() == 0:
    #         return np.max(self.readyTime, self.workcenter.getReadyTime())
    #     return np.max(self.readyTime, self.workcenter.getReadyTime())

    def setPriority(self, priority):
        self.priority = priority

    def priorTo(self, other):
        if self.priority < other.priority:
            return True
        elif self.priority > other.priority:
            return False
        return self.operation.getJob().getId() < other.operation.getJob().getId()

    def equals(self, other):
        return self.optionId == other.optionId and self.operation.getId() == other.operation.getId()

    def clone(self):
        clonedWC = self.workCenter.clone()
        clonedOp = OperationOption(self.operation, self.optionId, self.procTime, clonedWC)
        clonedOp.setReadyTime(self.readyTime)
        clonedOp.setWorkRemaining(self.workRemaining)
        clonedOp.setNumOpsRemaining(self.numOpsRemaining)
        clonedOp.setFlowDueDate(self.flowDueDate)
        clonedOp.setNextProcTime(self.nextProcTime)
        clonedOp.setPriority(self.priority)

        return clonedOp

    def __cmp__(self, other):
        if self.readyTime < other.readyTime:
            return -1

        if self.readyTime > other.readyTime:
            return 1

        return 0
