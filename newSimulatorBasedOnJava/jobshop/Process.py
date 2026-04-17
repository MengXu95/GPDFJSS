

class Process:

    def __init__(self, workCenter, machineId, operationOption, startTime, **kwargs):
        self.workCenter = workCenter
        self.machineId = machineId
        self.operationOption = operationOption
        self.startTime = startTime
        self.finishTime = startTime + operationOption.getProcTime()


    def getWorkCenter(self):
        return self.workCenter

    def getMachineId(self):
        return self.machineId

    def getOperationOption(self):
        return self.operationOption

    def getStartTime(self):
        return self.startTime

    def getFinishTime(self):
        return self.finishTime

    def getDuration(self):
        return self.finishTime - self.startTime

    def __cmp__(self, other):
        if self.startTime < other.startTime:
            return -1

        if self.startTime > other.startTime:
            return 1
    
        return 0