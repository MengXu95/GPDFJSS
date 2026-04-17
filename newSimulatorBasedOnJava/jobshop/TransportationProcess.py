

class TransportationProcess:
    def __init__(self, fromWorkCenter, toWorkCenter, operationOption, startTime, **kwargs):
        self.fromWorkCenter = fromWorkCenter
        self.toWorkCenter = toWorkCenter
        self.operationOption = operationOption
        self.startTime = startTime
        self.finishTime = startTime + operationOption.getOperation().getTransferTime()

    def getFromWorkcenter(self):
        return self.fromWorkCenter

    def getToWorkcenter(self):
        return self.toWorkCenter

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