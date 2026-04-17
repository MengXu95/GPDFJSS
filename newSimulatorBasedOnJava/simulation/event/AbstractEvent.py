

class AbstractEvent:
    def __init__(self, time, **kwargs):
        self.time = time


    def getTime(self):
        return self.time

    def getName(self):
        pass

    def trigger(self, simulation):
        pass

    def addSequencingDecisionSituation(self,simulation,situations,minQueueLength):
        pass

    def addRoutingDecisionSituation(self,simulation,situations,minOptions):
        pass

    def __cmp__(self, other):
        if self.time < other.time:
            return -1
        if self.time > other.time:
            return 1
        return 0

    